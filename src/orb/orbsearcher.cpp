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
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <functional>

#include <set>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <orbsearcher.h>
#include <messages.h>
#include <imageloader.h>

ORBSearcher::ORBSearcher(ORBIndex *index, ORBWordIndex *wordIndex)
    : index(index), wordIndex(wordIndex), orb(ORB::create(2000, 1.02, 100))
{ }


ORBSearcher::~ORBSearcher()
{ }


// Modern implementation using C++11 threading replaced RankingThread class


/**
 * @brief Processed a search request.
 * @param request the request to proceed.
 */
u_int32_t ORBSearcher::searchImage(SearchRequest &request)
{
    timeval t[3];
    gettimeofday(&t[0], NULL);

    cout << "Loading the image and extracting the ORBs." << endl;

    Mat img;
    u_int32_t i_ret = ImageLoader::loadImage(request.imageData.size(),
                                             request.imageData.data(), img);
    if (i_ret != OK)
        return i_ret;

    vector<KeyPoint> keypoints;
    Mat descriptors;

    orb->detectAndCompute(img, noArray(), keypoints, descriptors);

    gettimeofday(&t[1], NULL);

    cout << "time: " << getTimeDiff(t[0], t[1]) << " ms." << endl;
    cout << "Looking for the visual words. " << endl;

    const unsigned i_nbTotalIndexedImages = index->getTotalNbIndexedImages();
    const unsigned i_maxNbOccurences = i_nbTotalIndexedImages > 10000 ?
                                       0.15 * i_nbTotalIndexedImages
                                       : i_nbTotalIndexedImages;

    // Pre-check word occurrences to avoid unneeded lookups
    std::vector<bool> validWords(NB_VISUAL_WORDS, false);
    const unsigned i_maxOccurrences = i_maxNbOccurences; // Cache this value
    
    #pragma omp parallel for schedule(dynamic, 1000)
    for (u_int32_t i = 0; i < NB_VISUAL_WORDS; i++) {
        if (index->getWordNbOccurences(i) <= i_maxOccurrences) {
            validWords[i] = true;
        }
    }
    
    // Use vector instead of list for better cache locality
    std::unordered_map<u_int32_t, std::vector<Hit>> imageReqHits; // key: visual word, value: the found angles
    imageReqHits.reserve(std::min(keypoints.size(), size_t(5000))); // Reserve space to avoid rehashing
    
    constexpr int NB_NEIGHBORS = 1;
    
    // Process keypoints in batches for better memory access patterns
    const int batchSize = 64;
    for (unsigned i = 0; i < keypoints.size(); i += batchSize)
    {
        const unsigned endIdx = std::min<unsigned>(i + batchSize, keypoints.size());
        
        // Pre-allocate vectors for batch processing
        std::vector<std::vector<int>> batchIndices(endIdx - i, std::vector<int>(NB_NEIGHBORS));
        std::vector<std::vector<int>> batchDists(endIdx - i, std::vector<int>(NB_NEIGHBORS));

        // Perform batch kNN search
        for (unsigned j = i; j < endIdx; ++j) {
            wordIndex->knnSearch(descriptors.row(j), batchIndices[j-i], 
                                 batchDists[j-i], NB_NEIGHBORS);
        }

        // Process batch results
        for (unsigned j = i; j < endIdx; ++j) {
            for (int k = 0; k < NB_NEIGHBORS; ++k) {
                const unsigned i_wordId = batchIndices[j-i][k];
                
                // Skip invalid words using pre-computed validity
                if (!validWords[i_wordId])
                    continue;
                
                auto it = imageReqHits.find(i_wordId);
                if (it == imageReqHits.end()) {
                    // Convert the angle to a 16 bit integer.
                    Hit hit;
                    hit.i_imageId = 0;
                    hit.i_angle = keypoints[j].angle / 360 * (1 << 16);
                    hit.x = keypoints[j].pt.x;
                    hit.y = keypoints[j].pt.y;
                    
                    // Use emplace with hint for more efficient insertion
                    auto hint = imageReqHits.emplace(i_wordId, std::vector<Hit>{}).first;
                    hint->second.push_back(std::move(hit));
                }
            }
        }
    }

    gettimeofday(&t[2], NULL);
    cout << "time: " << getTimeDiff(t[1], t[2]) << " ms." << endl;

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
    std::unordered_map<u_int32_t, std::vector<Hit>> imageReqHits;
    u_int32_t i_ret = index->getImageWords(request.imageId, imageReqHits);

    if (i_ret != OK)
        return i_ret;

    gettimeofday(&t[1], NULL);
    cout << "time: " << getTimeDiff(t[0], t[1]) << " ms." << endl;

    return processSimilar(request, imageReqHits);
}


u_int32_t ORBSearcher::processSimilar(SearchRequest &request,
        std::unordered_map<u_int32_t, std::vector<Hit>> imageReqHits)
{
    timeval t[7];
    gettimeofday(&t[0], NULL);

    const unsigned i_nbTotalIndexedImages = index->getTotalNbIndexedImages();

    cout << imageReqHits.size() << " visual words kept for the request." << endl;
    cout << i_nbTotalIndexedImages << " images indexed in the index." << endl;

    // Preallocate space for index hits based on expected size
    std::unordered_map<u_int32_t, vector<Hit>> indexHits; // key: visual word id, values: index hits.
    indexHits.reserve(imageReqHits.size() * 1.25); // Add 25% for potential growth
    index->getImagesWithVisualWords(imageReqHits, indexHits);

    gettimeofday(&t[1], NULL);
    cout << "time: " << getTimeDiff(t[0], t[1]) << " ms." << endl;
    cout << "Ranking the images." << endl;

    index->readLock();
    
    // Use hardware concurrency to determine thread count
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const unsigned int NB_RANKING_THREAD = std::max(4u, hardware_threads);
    
    // Precalculate word weights to avoid redundant calculations
    std::unordered_map<u_int32_t, float> wordWeights;
    wordWeights.reserve(indexHits.size());
    
    for (const auto& pair : indexHits) {
        u_int32_t wordId = pair.first;
        const auto& hits = pair.second;
        wordWeights[wordId] = std::log((float)i_nbTotalIndexedImages / hits.size());
    }

    // Create a shared image weights map with atomic updates
    std::atomic<size_t> next_index(0);
    std::vector<u_int32_t> wordIds;
    wordIds.reserve(indexHits.size());
    
    for (const auto& pair : indexHits) {
        wordIds.push_back(pair.first);
    }
    
    // Use a concurrent map for weights
    std::unordered_map<u_int32_t, std::atomic<float>> sharedWeights;
    
    // Initialize all possible image IDs with zero scores
    std::vector<std::thread> threads;
    threads.reserve(NB_RANKING_THREAD);
    
    gettimeofday(&t[2], NULL);
    cout << "init threads time: " << getTimeDiff(t[1], t[2]) << " ms." << endl;
    
    // Create threads using C++11 thread support
    std::mutex weightsMutex;
    std::unordered_map<u_int32_t, float> weights; // final weights container
    weights.reserve(i_nbTotalIndexedImages / 2); // Reserve half the images for performance
    
    for (unsigned i = 0; i < NB_RANKING_THREAD; i++) {
        threads.emplace_back([&, i]() {
            // Local weights storage for this thread
            std::unordered_map<u_int32_t, float> localWeights;
            localWeights.reserve(i_nbTotalIndexedImages / NB_RANKING_THREAD);
            
            size_t idx;
            while ((idx = next_index.fetch_add(1)) < wordIds.size()) {
                u_int32_t wordId = wordIds[idx];
                const auto& hits = indexHits[wordId];
                const float wordWeight = wordWeights[wordId];
                
                for (const auto& hit : hits) {
                    unsigned imageId = hit.i_imageId;
                    unsigned totalWords = index->countTotalNbWord(imageId);
                    if (totalWords > 0) {
                        localWeights[imageId] += wordWeight / totalWords;
                    }
                }
            }
            
            // Now merge local weights into the shared weights under mutex protection
            std::lock_guard<std::mutex> lock(weightsMutex);
            for (const auto& pair : localWeights) {
                weights[pair.first] += pair.second;
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    gettimeofday(&t[3], NULL);
    cout << "compute time: " << getTimeDiff(t[2], t[3]) << " ms." << endl;
    cout << "reduce time: 0 ms." << endl; // Reduction already done in threads
    
    index->unlock();
    
    // Use a min-heap to efficiently track top results
    std::vector<SearchResult> topResults;
    topResults.reserve(300); // Only need the top 300 for reranking
    
    for (const auto& pair : weights) {
        if (topResults.size() < 300) {
            topResults.push_back(SearchResult(pair.second, pair.first, Rect()));
            // Convert to max-heap when we've collected 300 results
            if (topResults.size() == 300) {
                std::make_heap(topResults.begin(), topResults.end(), std::greater<SearchResult>());
            }
        } else if (pair.second > topResults.front().f_weight) {
            // Replace the smallest element and restore heap property
            std::pop_heap(topResults.begin(), topResults.end(), std::greater<SearchResult>());
            topResults.back() = SearchResult(pair.second, pair.first, Rect());
            std::push_heap(topResults.begin(), topResults.end(), std::greater<SearchResult>());
        }
    }
    
    // Convert to priority queue for compatibility with remaining code
    priority_queue<SearchResult> rankedResults;
    for (const auto& result : topResults) {
        rankedResults.push(result);
    }
    
    gettimeofday(&t[5], NULL);
    cout << "rankedResult time: " << getTimeDiff(t[4], t[5]) << " ms." << endl;
    cout << "Reranking 300 among " << weights.size() << " images." << endl;

    priority_queue<SearchResult> rerankedResults;
    reranker.rerank(imageReqHits, indexHits,
                    rankedResults, rerankedResults, 300);

    gettimeofday(&t[6], NULL);
    cout << "time: " << getTimeDiff(t[5], t[6]) << " ms." << endl;
    cout << "Returning the results. " << endl;

    returnResults(rerankedResults, request, 100);

    return SEARCH_RESULTS;
}


/**
 * @brief Return to the client the found results.
 * @param rankedResults the ranked list of results.
 * @param req the received search request.
 * @param i_maxNbResults the maximum number of results returned.
 */
void ORBSearcher::returnResults(priority_queue<SearchResult> &rankedResults,
                                SearchRequest &req, unsigned i_maxNbResults)
{
    // Pre-reserve memory to avoid reallocations
    req.results.reserve(i_maxNbResults);
    req.boundingRects.reserve(i_maxNbResults);
    req.scores.reserve(i_maxNbResults);
    req.tags.reserve(i_maxNbResults);
    
    // Batch process all results at once
    std::vector<u_int32_t> imageIds;
    imageIds.reserve(i_maxNbResults);
    
    unsigned i_res = 0;
    while(!rankedResults.empty() && i_res < i_maxNbResults)
    {
        const SearchResult &res = rankedResults.top();
        imageIds.push_back(res.i_imageId);
        
        i_res++;
        cout << "Id: " << res.i_imageId << ", score: " << res.f_weight << endl;
        
        req.results.push_back(res.i_imageId);
        req.boundingRects.push_back(res.boundingRect);
        req.scores.push_back(res.f_weight);

        rankedResults.pop();
    }
    
    // Batch process tags - minimize lock contention by doing this as a separate batch
    for (const auto& imageId : imageIds) {
        string tag;
        if (index->getTag(imageId, tag) == OK) {
            req.tags.push_back(std::move(tag));
        } else {
            req.tags.emplace_back("");
        }
    }
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
