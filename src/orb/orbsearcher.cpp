/*****************************************************************************
 * Copyright (C) 2014 Visualink
 *
 * Authors: Adrien Maglo <adrien@visualink.io>
 *
 * This file is part of Pastec.
 *
 * Pastec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Pastec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Pastec.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include <iostream>
#include <fstream>
#include <sys/time.h>

#include <set>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>  // For std::partial_sort

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <orbsearcher.h>
#include <messages.h>
#include <imageloader.h>

// C++17 doesn't need tr1 namespace anymore

ORBSearcher::ORBSearcher(ORBIndex *index, ORBWordIndex *wordIndex)
    : index(index), wordIndex(wordIndex), orb(ORB::create(2000, 1.02, 100)),
      threadPool(NUM_THREADS)  // Initialize thread pool once
{
    // Pre-compute word counts are already stored in the index
    
    // Initialize thread-specific word indices with shared words
    for (int i = 0; i < NUM_THREADS; i++) {
        threadWordIndices.push_back(std::make_unique<ORBWordIndex>(wordIndex->getWords()));
    }
}


/**
 * @brief Process a batch of words for TF-IDF computation
 * @param batch The batch of words to process
 * @param wordCounts Vector of word counts per image
 * @param i_nbTotalIndexedImages Total number of indexed images
 * @param maxImageId Maximum image ID
 * @return Vector of weights for each image
 */
vector<float> ORBSearcher::processTFIDFBatch(
    const vector<pair<u_int32_t, const vector<Hit>*>>& batch,
    const vector<unsigned>& wordCounts,
    unsigned i_nbTotalIndexedImages,
    unsigned maxImageId)
{
    // Create a local weights vector for this batch
    vector<float> batchWeights(maxImageId + 1, 0.0f);
    
    // Process each word in the batch
    for (const auto& wordPair : batch) {
        const u_int32_t wordId = wordPair.first;
        const vector<Hit>* hits = wordPair.second;
        
        // Calculate IDF weight for this word
        const float f_weight = log((float)i_nbTotalIndexedImages / hits->size());
        
        // Update weights for all images containing this word
        for (const Hit& hit : *hits) {
            // TF-IDF calculation
            unsigned i_totalNbWords = wordCounts[hit.i_imageId];
            batchWeights[hit.i_imageId] += f_weight / i_totalNbWords;
        }
    }
    
    return batchWeights;
}


ORBSearcher::~ORBSearcher()
{
    threadPool.join();  // Ensure all tasks complete before destruction
}


// Process a batch of keypoints
std::unordered_map<u_int32_t, list<Hit>> ORBSearcher::processKeyPointBatch(
    const Mat& descriptors,
    const vector<KeyPoint>& keypoints,
    size_t startIdx,
    size_t endIdx,
    ORBWordIndex* localWordIndex)
{
    const unsigned i_nbTotalIndexedImages = index->getTotalNbIndexedImages();
    const unsigned i_maxNbOccurences = i_nbTotalIndexedImages > 10000 ?
                                      0.15 * i_nbTotalIndexedImages
                                      : i_nbTotalIndexedImages;

    // Create a local results map for this thread
    std::unordered_map<u_int32_t, list<Hit>> localResults;

    for (unsigned i = startIdx; i < endIdx; ++i)
    {
        #define NB_NEIGHBORS 1

        vector<int> indices(NB_NEIGHBORS);
        vector<int> dists(NB_NEIGHBORS);
        
        localWordIndex->knnSearch(descriptors.row(i), indices, dists, NB_NEIGHBORS);

        for (unsigned j = 0; j < indices.size(); ++j)
        {
            const unsigned i_wordId = indices[j];

            if (index->getWordNbOccurences(i_wordId) > i_maxNbOccurences)
                continue;
            
            if (localResults.find(i_wordId) == localResults.end())
            {
                // Convert the angle to a 16 bit integer.
                Hit hit;
                hit.i_imageId = 0;
                hit.i_angle = keypoints[i].angle / 360 * (1 << 16);
                hit.x = keypoints[i].pt.x;
                hit.y = keypoints[i].pt.y;

                localResults[i_wordId].push_back(hit);
            }
        }
    }
    
    // Return the local results
    return localResults;
}


// Helper function for sift-down operation in min-heap
static void siftDown(std::pair<float, u_int32_t>* heap, size_t size, size_t idx) {
    size_t smallest = idx;
    size_t left = 2 * idx + 1;
    size_t right = 2 * idx + 2;
    
    if (left < size && heap[left].first < heap[smallest].first)
        smallest = left;
        
    if (right < size && heap[right].first < heap[smallest].first)
        smallest = right;
        
    if (smallest != idx) {
        std::swap(heap[idx], heap[smallest]);
        siftDown(heap, size, smallest);
    }
}


/**
 * @brief Processed a search request.
 * @param request the request to proceed.
 */
u_int32_t ORBSearcher::searchImage(SearchRequest &request)
{
    // Add timing at the very beginning of the function
    timeval t_entry, t_feature_start;
    gettimeofday(&t_entry, NULL);
    cout << "Entering searchImage function." << endl;
    
    timeval t[3];
    gettimeofday(&t_feature_start, NULL);
    t[0] = t_feature_start;

    cout << "Loading the image and extracting the ORBs." << endl;

    // Time the image loading operation
    timeval t_load_start, t_load_end;
    gettimeofday(&t_load_start, NULL);
    
    Mat img;
    u_int32_t i_ret = ImageLoader::loadImage(request.imageData.size(),
                                             request.imageData.data(), img);
    if (i_ret != OK)
        return i_ret;
        
    gettimeofday(&t_load_end, NULL);
    cout << "Image loading time: " << getTimeDiff(t_load_start, t_load_end) << " ms." << endl;

    // Time the feature extraction operation
    timeval t_extract_start, t_extract_end;
    gettimeofday(&t_extract_start, NULL);
    
    vector<KeyPoint> keypoints;
    Mat descriptors;

    orb->detectAndCompute(img, noArray(), keypoints, descriptors);
    
    gettimeofday(&t_extract_end, NULL);
    cout << "ORB feature extraction time: " << getTimeDiff(t_extract_start, t_extract_end) << " ms." << endl;

    gettimeofday(&t[1], NULL);

    cout << "time: " << getTimeDiff(t[0], t[1]) << " ms." << endl;
    cout << "Initial setup time (before feature extraction): " << getTimeDiff(t_entry, t_feature_start) << " ms." << endl;
    cout << "Looking for the visual words. " << endl;

    const unsigned i_nbTotalIndexedImages = index->getTotalNbIndexedImages();
    const unsigned i_maxNbOccurences = i_nbTotalIndexedImages > 10000 ?
                                       0.15 * i_nbTotalIndexedImages
                                       : i_nbTotalIndexedImages;

    // Time the visual word extraction loop
    timeval t_word_start, t_word_end;
    gettimeofday(&t_word_start, NULL);
    
    std::unordered_map<u_int32_t, list<Hit> > imageReqHits; // key: visual word, value: the found angles
    
    // Time the parallel knnSearch operations
    timeval t_knn_total_start, t_knn_total_end;
    gettimeofday(&t_knn_total_start, NULL);
    
    // Use the persistent thread pool
    // boost::asio::thread_pool pool(NUM_THREADS);
    
    // Calculate batch size based on FEATURE_BATCH_COUNT
    size_t totalKeypoints = keypoints.size();
    size_t batchSize = (totalKeypoints + FEATURE_BATCH_COUNT - 1) / FEATURE_BATCH_COUNT; // Ceiling division
    
    cout << "Processing " << totalKeypoints << " keypoints in " << FEATURE_BATCH_COUNT 
         << " batches with batch size " << batchSize << endl;
    
    // Create a vector to hold futures for each task
    std::vector<std::future<std::unordered_map<u_int32_t, list<Hit>>>> futures;
    
    // Submit tasks to the thread pool
    for (int b = 0; b < FEATURE_BATCH_COUNT; b++) {
        size_t startIdx = b * batchSize;
        size_t endIdx = std::min(startIdx + batchSize, totalKeypoints);
        
        // Skip empty batches
        if (startIdx >= totalKeypoints) {
            continue;
        }
        
        cout << "Batch " << b << " processing keypoints " << startIdx << " to " << endIdx - 1 << endl;
        
        // Create a packaged task that returns a results map
        auto task = std::make_shared<std::packaged_task<std::unordered_map<u_int32_t, list<Hit>>()>>(
            [this, &descriptors, &keypoints, startIdx, endIdx, b]() {
                return this->processKeyPointBatch(descriptors, keypoints, startIdx, endIdx, threadWordIndices[b % NUM_THREADS].get());
            }
        );
        
        // Get the future from the task
        futures.push_back(task->get_future());
        
        // Submit the task to the thread pool
        boost::asio::post(threadPool, [task]() { (*task)(); });
    }
    
    // Wait for all tasks to complete and merge their results
    for (auto& future : futures) {
        auto threadResults = future.get();
        
        // Merge thread results into the final results map
        for (auto& [wordId, hits] : threadResults) {
            if (imageReqHits.find(wordId) == imageReqHits.end()) {
                imageReqHits[wordId] = std::move(hits);
            } else {
                // If the word already exists, append the hits
                imageReqHits[wordId].splice(imageReqHits[wordId].end(), std::move(hits));
            }
        }
    }
    
    // We don't join the persistent thread pool here, it will be joined in the destructor
    // threadPool.join();
    
    gettimeofday(&t_knn_total_end, NULL);
    unsigned long total_knn_time = getTimeDiff(t_knn_total_start, t_knn_total_end);
    cout << "Total parallel knnSearch operations time: " << total_knn_time << " ms." << endl;
    cout << "Average time per keypoint: " << (keypoints.size() > 0 ? (float)total_knn_time / keypoints.size() : 0) << " ms." << endl;
    
    gettimeofday(&t_word_end, NULL);
    cout << "Visual word extraction loop time: " << getTimeDiff(t_word_start, t_word_end) << " ms." << endl;

    gettimeofday(&t[2], NULL);
    cout << "time: " << getTimeDiff(t[1], t[2]) << " ms." << endl;

    // Add timing before calling processSimilar
    timeval t_before_process;
    gettimeofday(&t_before_process, NULL);
    cout << "Time between visual word extraction and processSimilar: " << getTimeDiff(t[2], t_before_process) << " ms." << endl;

    return processSimilar(request, imageReqHits);
}


/**
 * @brief Processed a similarity request.
 * @param request the request to proceed.
 */
u_int32_t ORBSearcher::searchSimilar(SearchRequest &request)
{
    timeval t[2];
    gettimeofday(&t[0], NULL);

    cout << "Loading the image words from the index." << endl;

    // key: visual word, value: the found angles
    std::unordered_map<u_int32_t, list<Hit> > imageReqHits;
    u_int32_t i_ret = index->getImageWords(request.imageId, imageReqHits);

    if (i_ret != OK)
        return i_ret;

    gettimeofday(&t[1], NULL);
    cout << "time: " << getTimeDiff(t[0], t[1]) << " ms." << endl;

    return processSimilar(request, imageReqHits);
}


u_int32_t ORBSearcher::processSimilar(SearchRequest &request,
        std::unordered_map<u_int32_t, list<Hit> > imageReqHits)
{
    timeval t[7];
    gettimeofday(&t[0], NULL);

    cout << "Processing similar with " << imageReqHits.size() << " visual words in query." << endl;

    // Add timing for initialization before index lookup
    timeval t_init_start;
    gettimeofday(&t_init_start, NULL);
    cout << "Time between entering processSimilar and starting initialization: " << getTimeDiff(t[0], t_init_start) << " ms." << endl;

    const unsigned i_nbTotalIndexedImages = index->getTotalNbIndexedImages();

    std::unordered_map<u_int32_t, const vector<Hit>* > indexHits; // key: visual word id, values: index hits.
    indexHits.rehash(imageReqHits.size());
    
    // Add timing before actual index lookup
    timeval t_before_lookup;
    gettimeofday(&t_before_lookup, NULL);
    cout << "Initialization time before index lookup: " << getTimeDiff(t_init_start, t_before_lookup) << " ms." << endl;
    
    // Time the actual index lookup operation
    timeval t_lookup_start, t_lookup_end;
    gettimeofday(&t_lookup_start, NULL);
    
    index->getImagesWithVisualWords(imageReqHits, indexHits);
    
    gettimeofday(&t_lookup_end, NULL);
    cout << "Actual index lookup operation time: " << getTimeDiff(t_lookup_start, t_lookup_end) << " ms." << endl;

    // Time the hit counting operation
    timeval t_count_start, t_count_end;
    gettimeofday(&t_count_start, NULL);
    
    // Count total hits across all visual words
    unsigned totalHits = 0;
    unsigned maxHitsPerWord = 0;
    unsigned wordsWithNoHits = 0;
    
    for (auto it = indexHits.begin(); it != indexHits.end(); ++it) {
        unsigned wordHits = it->second->size();
        totalHits += wordHits;
        
        if (wordHits > maxHitsPerWord)
            maxHitsPerWord = wordHits;
            
        if (wordHits == 0)
            wordsWithNoHits++;
    }
    
    gettimeofday(&t_count_end, NULL);
    cout << "Hit counting operation time: " << getTimeDiff(t_count_start, t_count_end) << " ms." << endl;

    gettimeofday(&t[1], NULL);
    cout << "Index lookup time: " << getTimeDiff(t[0], t[1]) << " ms." << endl;
    cout << "Found " << indexHits.size() << " visual words in index out of " << imageReqHits.size() << " requested." << endl;
    cout << "Total hits: " << totalHits << ", avg hits per word: " << (indexHits.size() > 0 ? totalHits / indexHits.size() : 0) << endl;
    cout << "Max hits per word: " << maxHitsPerWord << ", words with no hits: " << wordsWithNoHits << endl;
    cout << "Ranking the images." << endl;

    gettimeofday(&t[2], NULL);
    cout << "Single-threaded ranking initialization time: " << getTimeDiff(t[1], t[2]) << " ms." << endl;

    // Get the maximum image ID and word counts
    const unsigned maxImageId = index->getWordCountVector().size() - 1;
    const vector<unsigned>& wordCounts = index->getWordCountVector();
    
    // Create a pre-allocated array for direct indexing of weights
    vector<float> weights(maxImageId + 1, 0.0f);
    
    // Process all visual words in parallel
    unsigned totalHitsProcessed = 0;
    
    // Time the weight computation loop
    timeval t_weight_loop_start, t_weight_loop_end;
    gettimeofday(&t_weight_loop_start, NULL);
    
    // Convert the map to a vector for easier batch division
    vector<pair<u_int32_t, const vector<Hit>*>> wordPairs;
    wordPairs.reserve(indexHits.size());
    
    for (const auto& pair : indexHits) {
        wordPairs.push_back({pair.first, pair.second});
        totalHitsProcessed += pair.second->size();
    }
    
    // Calculate batch size based on WEIGHT_BATCH_COUNT
    size_t totalWords = wordPairs.size();
    size_t batchSize = (totalWords + WEIGHT_BATCH_COUNT - 1) / WEIGHT_BATCH_COUNT; // Ceiling division
    
    cout << "Processing " << totalWords << " words in " << WEIGHT_BATCH_COUNT 
         << " batches with batch size " << batchSize << endl;
    
    // Create a vector to hold futures for each task
    std::vector<std::future<vector<float>>> futures;
    
    // Submit tasks to the thread pool
    for (int b = 0; b < WEIGHT_BATCH_COUNT; b++) {
        size_t startIdx = b * batchSize;
        size_t endIdx = std::min(startIdx + batchSize, totalWords);
        
        // Skip empty batches
        if (startIdx >= totalWords) {
            continue;
        }
        
        cout << "Batch " << b << " processing words " << startIdx << " to " << endIdx - 1 << endl;
        
        // Create the batch
        vector<pair<u_int32_t, const vector<Hit>*>> batch(
            wordPairs.begin() + startIdx,
            wordPairs.begin() + endIdx
        );
        
        // Create a packaged task that returns a weights vector
        auto task = std::make_shared<std::packaged_task<vector<float>()>>(
            [this, batch, &wordCounts, i_nbTotalIndexedImages, maxImageId]() {
                return this->processTFIDFBatch(batch, wordCounts, i_nbTotalIndexedImages, maxImageId);
            }
        );
        
        // Get the future from the task
        futures.push_back(task->get_future());
        
        // Submit the task to the thread pool
        boost::asio::post(threadPool, [task]() { (*task)(); });
    }
    
    // Wait for all tasks to complete and merge their results
    for (auto& future : futures) {
        auto batchWeights = future.get();
        
        // Merge batch weights into the final weights vector
        for (u_int32_t id = 0; id <= maxImageId; ++id) {
            weights[id] += batchWeights[id];
        }
    }
    
    gettimeofday(&t_weight_loop_end, NULL);
    cout << "Parallel TF-IDF weight computation time: " << getTimeDiff(t_weight_loop_start, t_weight_loop_end) << " ms." << endl;

    gettimeofday(&t[3], NULL);
    cout << "Weight computation time: " << getTimeDiff(t[2], t[3]) << " ms." << endl;
    cout << "Total hits processed: " << totalHitsProcessed << endl;
    
    // Add timing before top-N selection
    timeval t_before_topn;
    gettimeofday(&t_before_topn, NULL);
    cout << "Time between weight computation and top-N selection: " << getTimeDiff(t[3], t_before_topn) << " ms." << endl;
    
    // Time the heap operations
    timeval t_heap_start, t_heap_end, t_sort_start, t_sort_end;
    gettimeofday(&t_heap_start, NULL);
    
    // Find top 300 results using a bounded min-heap (keeps largest elements by replacing smallest)
    const unsigned TOP_N = 300;
    std::pair<float, u_int32_t> topResults[TOP_N];
    size_t heapSize = 0;
    
    // Process all images in a single pass
    for (u_int32_t id = 0; id <= maxImageId; ++id) {
        if (weights[id] > 0) {
            if (heapSize < TOP_N) {
                // Heap not full yet, just add the element
                topResults[heapSize++] = {weights[id], id};
                
                // If we just filled the heap, heapify it once
                if (heapSize == TOP_N) {
                    // Build min-heap (smallest element at root)
                    for (int i = heapSize / 2 - 1; i >= 0; i--) {
                        siftDown(topResults, heapSize, i);
                    }
                }
            } 
            else if (weights[id] > topResults[0].first) {
                // Heap is full and we found a larger weight
                // Replace the smallest element (root) and sift down
                topResults[0] = {weights[id], id};
                siftDown(topResults, heapSize, 0);
            }
        }
    }
    
    gettimeofday(&t_heap_end, NULL);
    cout << "Heap construction and maintenance time: " << getTimeDiff(t_heap_start, t_heap_end) << " ms." << endl;
    
    // Time the sorting operation
    gettimeofday(&t_sort_start, NULL);
    
    // Convert heap to sorted vector (descending order by weight)
    vector<pair<float, u_int32_t>> sortedResults(topResults, topResults + heapSize);
    std::sort(sortedResults.begin(), sortedResults.end(), 
              [](const std::pair<float, u_int32_t>& a, const std::pair<float, u_int32_t>& b) { 
                  return a.first > b.first; 
              });
              
    gettimeofday(&t_sort_end, NULL);
    cout << "Sorting time: " << getTimeDiff(t_sort_start, t_sort_end) << " ms." << endl;

    gettimeofday(&t[5], NULL);
    cout << "Top-" << TOP_N << " selection time: " << getTimeDiff(t[3], t[5]) << " ms." << endl;
    cout << "Reranking " << sortedResults.size() << " images." << endl;
    
    // Debug: Print top 5 weights to verify we're getting the largest weights
    cout << "Top 5 weights: ";
    for (unsigned i = 0; i < std::min(5u, (unsigned)sortedResults.size()); ++i) {
        cout << sortedResults[i].first << " (id: " << sortedResults[i].second << ") ";
    }
    cout << endl;

    // Check if forward index is available and use the optimized reranking method
    vector<SearchResult> rerankedResults;
    ORBIndex* orbIndex = static_cast<ORBIndex*>(index);
    
    // Time the reranking preparation
    timeval t_rerank_prep_start, t_rerank_prep_end;
    gettimeofday(&t_rerank_prep_start, NULL);
    
    if (orbIndex->hasForwardIndex()) {
        // Get the set of image IDs to rerank
        unordered_set<u_int32_t> firstImageIds;
        for (unsigned i = 0; i < min(TOP_N, (unsigned)sortedResults.size()); i++) {
            firstImageIds.insert(sortedResults[i].second);
        }
        
        gettimeofday(&t_rerank_prep_end, NULL);
        cout << "Reranking preparation time: " << getTimeDiff(t_rerank_prep_start, t_rerank_prep_end) << " ms." << endl;
        
        cout << "Using forward index for reranking." << endl;
        
        // Time the actual reranking operation
        timeval t_rerank_start, t_rerank_end;
        gettimeofday(&t_rerank_start, NULL);
        
        // Use the forward index reranking
        rerankedResults = reranker.rerankUsingForwardIndex(imageReqHits, orbIndex, firstImageIds);
        
        gettimeofday(&t_rerank_end, NULL);
        cout << "Forward index reranking operation time: " << getTimeDiff(t_rerank_start, t_rerank_end) << " ms." << endl;
    } else {
        // Fall back to the original reranking
        unordered_set<u_int32_t> firstImageIds;
        for (unsigned i = 0; i < min(TOP_N, (unsigned)sortedResults.size()); i++) {
            firstImageIds.insert(sortedResults[i].second);
        }
        
        gettimeofday(&t_rerank_prep_end, NULL);
        cout << "Reranking preparation time: " << getTimeDiff(t_rerank_prep_start, t_rerank_prep_end) << " ms." << endl;
        
        cout << "Forward index not available, using standard reranking." << endl;
        
        // Time the actual reranking operation
        timeval t_rerank_start, t_rerank_end;
        gettimeofday(&t_rerank_start, NULL);
        
        rerankedResults = reranker.rerank(imageReqHits, indexHits, sortedResults, TOP_N);
        
        gettimeofday(&t_rerank_end, NULL);
        cout << "Standard reranking operation time: " << getTimeDiff(t_rerank_start, t_rerank_end) << " ms." << endl;
    }

    gettimeofday(&t[6], NULL);
    cout << "Reranking time: " << getTimeDiff(t[5], t[6]) << " ms." << endl;
    cout << "Returning the results. " << endl;

    // Add timing for result preparation
    timeval t_before_results, t_after_results;
    gettimeofday(&t_before_results, NULL);
    cout << "Time between reranking and result preparation: " << getTimeDiff(t[6], t_before_results) << " ms." << endl;
    
    returnResults(rerankedResults, request, 100);
    
    gettimeofday(&t_after_results, NULL);
    cout << "Result preparation time: " << getTimeDiff(t_before_results, t_after_results) << " ms." << endl;

    return SEARCH_RESULTS;
}


/**
 * @brief Return to the client the found results.
 * @param rankedResults the ranked list of results.
 * @param req the received search request.
 * @param i_maxNbResults the maximum number of results returned.
 */
void ORBSearcher::returnResults(vector<SearchResult> &rankedResults,
                              SearchRequest &req, unsigned i_maxNbResults)
{
    list<u_int32_t> imageIds;

    cout << "Number of reranked results: " << rankedResults.size() << endl;
    
    unsigned i_res = 0;
    for (const auto& res : rankedResults)
    {
        if (i_res >= i_maxNbResults)
            break;
            
        imageIds.push_back(res.i_imageId);
        i_res++;
        cout << "Id: " << res.i_imageId << ", score: " << res.f_weight << endl;
        req.results.push_back(res.i_imageId);
        req.boundingRects.push_back(res.boundingRect);
        req.scores.push_back(res.f_weight);

        string tag;
        if (index->getTag(res.i_imageId, tag) == OK)
            req.tags.push_back(tag);
        else
            req.tags.push_back("");
    }
    
    cout << "Total results returned: " << i_res << endl;
}


/**
 * @brief Get the time difference in ms between two instants.
 * @param t1
 * @param t2
 */
unsigned long ORBSearcher::getTimeDiff(const timeval t1, const timeval t2) const
{
    return ((t2.tv_sec - t1.tv_sec) * 1000000
            + (t2.tv_usec - t1.tv_usec)) / 1000;
}
