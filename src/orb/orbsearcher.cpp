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
#ifndef __APPLE__
#include <tr1/unordered_set>
#include <tr1/unordered_map>
#else
#include <unordered_set>
#include <unordered_map>
#endif
#include <queue>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <orbsearcher.h>
#include <messages.h>
#include <imageloader.h>

#ifndef __APPLE__
using namespace std::tr1;
#endif

ORBSearcher::ORBSearcher(ORBIndex *index, ORBWordIndex *wordIndex)
    : index(index), wordIndex(wordIndex), orb(ORB::create(2000, 1.02, 100))
{
    // Pre-compute word counts are already stored in the index
}


ORBSearcher::~ORBSearcher()
{ }


/**
 * @brief The RankingThread class
 * This threads computes the tf-idf weights of the images that contains the words
 * given in argument.
 */
class RankingThread : public Thread
{
public:
    RankingThread(ORBIndex *index, const unsigned i_nbTotalIndexedImages,
                  std::unordered_map<u_int32_t, const vector<Hit>* > &indexHits)
        : index(index), i_nbTotalIndexedImages(i_nbTotalIndexedImages),
          indexHits(indexHits), wordCounts(index->getWordCountVector()) 
    {
        // Reserve some initial capacity to avoid frequent reallocations
        weightVec.reserve(1000);
    }

    void addWord(u_int32_t i_wordId)
    {
        wordIds.push_back(i_wordId);
    }

    void *run()
    {
        timeval t_start, t_init, t_end;
        gettimeofday(&t_start, NULL);
        
        // Use a temporary map for accumulating weights
        std::unordered_map<u_int32_t, float> tempWeights;
        tempWeights.rehash(wordIds.size() * 10); // Estimate size based on word count
        
        gettimeofday(&t_init, NULL);
        cout << "[RankingThread] Init time: " << getTimeDiff(t_start, t_init) << " ms." << endl;
        
        unsigned totalHitsProcessed = 0;

        for (deque<u_int32_t>::const_iterator it = wordIds.begin();
            it != wordIds.end(); ++it)
        {
            const vector<Hit> *hits = indexHits[*it];
            totalHitsProcessed += hits->size();

            const float f_weight = log((float)i_nbTotalIndexedImages / hits->size());

            for (vector<Hit>::const_iterator it2 = hits->begin();
                 it2 != hits->end(); ++it2)
            {
                /* TF-IDF according to the paper "Video Google:
                 * A Text Retrieval Approach to Object Matching in Videos" */
                // Direct access to word counts without function call or bounds checking
                unsigned i_totalNbWords = wordCounts[it2->i_imageId];
                tempWeights[it2->i_imageId] += f_weight / i_totalNbWords;
            }
        }
        
        // Convert map to vector only once at the end
        weightVec.reserve(tempWeights.size());
        for (const auto& pair : tempWeights) {
            weightVec.emplace_back(pair.second, pair.first); // weight first for sorting
        }
        
        gettimeofday(&t_end, NULL);
        cout << "[RankingThread] Processing time: " << getTimeDiff(t_init, t_end) << " ms." << endl;
        cout << "[RankingThread] Total time: " << getTimeDiff(t_start, t_end) << " ms." << endl;
        cout << "[RankingThread] Words processed: " << wordIds.size() << endl;
        cout << "[RankingThread] Total hits processed: " << totalHitsProcessed << endl;
        cout << "[RankingThread] Unique images weighted: " << weightVec.size() << endl;

        return NULL;
    }
    
    unsigned long getTimeDiff(const timeval t1, const timeval t2) const
    {
        return ((t2.tv_sec - t1.tv_sec) * 1000000
                + (t2.tv_usec - t1.tv_usec)) / 1000;
    }

    ORBIndex *index;
    const unsigned i_nbTotalIndexedImages;
    std::unordered_map<u_int32_t, const vector<Hit>* > &indexHits;
    deque<u_int32_t> wordIds;
    vector<pair<float, u_int32_t>> weightVec; // weight, image id pairs for sorting
    const vector<unsigned>& wordCounts; // Direct reference to word counts
};


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

    // No locks needed since the index is read-only during queries
    #define NB_RANKING_THREAD 20

    // Map the ranking to threads.
    unsigned i_wordsPerThread = indexHits.size() / NB_RANKING_THREAD + 1;
    RankingThread *threads[NB_RANKING_THREAD];

    std::unordered_map<u_int32_t, const vector<Hit>* >::const_iterator it = indexHits.begin();
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
    {
        threads[i] = new RankingThread(index, i_nbTotalIndexedImages, indexHits);

        unsigned i_nbWords = 0;
        for (; it != indexHits.end() && i_nbWords < i_wordsPerThread; ++it, ++i_nbWords)
            threads[i]->addWord(it->first);
    }

    gettimeofday(&t[2], NULL);
    cout << "Thread initialization time: " << getTimeDiff(t[1], t[2]) << " ms." << endl;

    // Compute
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
        threads[i]->start();
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
        threads[i]->join();

    gettimeofday(&t[3], NULL);
    cout << "Thread computation time: " << getTimeDiff(t[2], t[3]) << " ms." << endl;

    // Estimate the total size needed for the combined vector
    size_t totalSize = 0;
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i) {
        totalSize += threads[i]->weightVec.size();
    }
    
    // Combine all thread results into a single vector
    vector<pair<float, u_int32_t>> allWeights;
    allWeights.reserve(totalSize);
    
    // Use a map to combine weights for the same image ID
    std::unordered_map<u_int32_t, float> combinedWeights;
    combinedWeights.reserve(totalSize);
    
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i) {
        for (const auto& pair : threads[i]->weightVec) {
            combinedWeights[pair.second] += pair.first;
        }
    }
    
    // Convert the combined weights to a vector
    allWeights.reserve(combinedWeights.size());
    for (const auto& pair : combinedWeights) {
        allWeights.emplace_back(pair.second, pair.first); // weight, imageId
    }

    gettimeofday(&t[4], NULL);
    cout << "Result reduction time: " << getTimeDiff(t[3], t[4]) << " ms." << endl;
    cout << "Total unique images found: " << allWeights.size() << endl;

    // Free the memory
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
        delete threads[i];

    // Use partial sort - much faster than full sort for large datasets
    const unsigned TOP_N = 300; // Only need top 300 for reranking
    std::partial_sort(allWeights.begin(), 
                     allWeights.begin() + std::min(TOP_N, (unsigned)allWeights.size()),
                     allWeights.end(),
                     [](const std::pair<float, u_int32_t>& a, const std::pair<float, u_int32_t>& b) { 
                         return a.first > b.first; 
                     });

    gettimeofday(&t[5], NULL);
    cout << "Partial sort time: " << getTimeDiff(t[4], t[5]) << " ms." << endl;
    cout << "Reranking 300 among " << allWeights.size() << " images." << endl;

    // Rerank using the vector directly
    vector<SearchResult> rerankedResults = reranker.rerank(imageReqHits, indexHits, allWeights, 300);

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
