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
    : index(index), wordIndex(wordIndex), orb(ORB::create(2000, 1.02, 100))
{
    // Pre-compute word counts are already stored in the index
}


ORBSearcher::~ORBSearcher()
{ }


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

    std::unordered_map<u_int32_t, list<Hit> > imageReqHits; // key: visual word, value: the found angles
    for (unsigned i = 0; i < keypoints.size(); ++i)
    {
        #define NB_NEIGHBORS 1

        vector<int> indices(NB_NEIGHBORS);
        vector<int> dists(NB_NEIGHBORS);
        wordIndex->knnSearch(descriptors.row(i), indices,
                           dists, NB_NEIGHBORS);

        for (unsigned j = 0; j < indices.size(); ++j)
        {
            const unsigned i_wordId = indices[j];

            if (index->getWordNbOccurences(i_wordId) > i_maxNbOccurences)
                continue;

            if (imageReqHits.find(i_wordId) == imageReqHits.end())
            {
                // Convert the angle to a 16 bit integer.
                Hit hit;
                hit.i_imageId = 0;
                hit.i_angle = keypoints[i].angle / 360 * (1 << 16);
                hit.x = keypoints[i].pt.x;
                hit.y = keypoints[i].pt.y;

                imageReqHits[i_wordId].push_back(hit);
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

    const unsigned i_nbTotalIndexedImages = index->getTotalNbIndexedImages();

    std::unordered_map<u_int32_t, const vector<Hit>* > indexHits; // key: visual word id, values: index hits.
    indexHits.rehash(imageReqHits.size());
    index->getImagesWithVisualWords(imageReqHits, indexHits);

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
    
    // Process all visual words in a single loop
    unsigned totalHitsProcessed = 0;
    
    // Compute TF-IDF weights for all images
    for (auto it = indexHits.begin(); it != indexHits.end(); ++it) {
        const u_int32_t wordId = it->first;
        const vector<Hit>* hits = it->second;
        totalHitsProcessed += hits->size();
        
        // Calculate IDF weight for this word
        const float f_weight = log((float)i_nbTotalIndexedImages / hits->size());
        
        // Update weights for all images containing this word
        for (const Hit& hit : *hits) {
            // TF-IDF calculation
            unsigned i_totalNbWords = wordCounts[hit.i_imageId];
            weights[hit.i_imageId] += f_weight / i_totalNbWords;
        }
    }

    gettimeofday(&t[3], NULL);
    cout << "Weight computation time: " << getTimeDiff(t[2], t[3]) << " ms." << endl;
    cout << "Total hits processed: " << totalHitsProcessed << endl;
    
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
    
    // Convert heap to sorted vector (descending order by weight)
    vector<pair<float, u_int32_t>> sortedResults(topResults, topResults + heapSize);
    std::sort(sortedResults.begin(), sortedResults.end(), 
              [](const std::pair<float, u_int32_t>& a, const std::pair<float, u_int32_t>& b) { 
                  return a.first > b.first; 
              });

    gettimeofday(&t[5], NULL);
    cout << "Top-" << TOP_N << " selection time: " << getTimeDiff(t[3], t[5]) << " ms." << endl;
    cout << "Reranking " << sortedResults.size() << " images." << endl;
    
    // Debug: Print top 5 weights to verify we're getting the largest weights
    cout << "Top 5 weights: ";
    for (unsigned i = 0; i < std::min(5u, (unsigned)sortedResults.size()); ++i) {
        cout << sortedResults[i].first << " (id: " << sortedResults[i].second << ") ";
    }
    cout << endl;

    // Rerank using the vector directly
    vector<SearchResult> rerankedResults = reranker.rerank(imageReqHits, indexHits, sortedResults, TOP_N);

    gettimeofday(&t[6], NULL);
    cout << "Reranking time: " << getTimeDiff(t[5], t[6]) << " ms." << endl;
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
