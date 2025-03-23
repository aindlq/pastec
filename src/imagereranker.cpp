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
#include <cassert>
#include <math.h>

#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <imagereranker.h>


void *RANSACThread::run()
{
    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);
    
    unsigned ransacAttempts = 0;
    unsigned successfulRansacs = 0;
    unsigned skippedDueToLowValue = 0;
    unsigned skippedDueToFewPoints = 0;
    unsigned skippedDueToZeroH = 0;
    
    for (unsigned i = 0; i < imageIds.size(); ++i)
    {
        const unsigned i_imageId = imageIds[i];
        const Histogram histogram = histograms[i];
        unsigned i_binMax = max_element(histogram.bins, histogram.bins + HISTOGRAM_NB_BINS) - histogram.bins;
        float i_maxVal = histogram.bins[i_binMax];
        
        if (i_maxVal > 10)
        {
            RANSACTask &task = imgTasks[i_imageId];
            assert(task.points1.size() == task.points2.size());

            if (task.points1.size() >= RANSAC_MIN_INLINERS)
            {
                ransacAttempts++;
                struct timeval t_ransac_start, t_ransac_end;
                gettimeofday(&t_ransac_start, NULL);
                
                Mat H = pastecEstimateRigidTransform(task.points2, task.points1, true);
                
                gettimeofday(&t_ransac_end, NULL);
                unsigned long ransac_time = ((t_ransac_end.tv_sec - t_ransac_start.tv_sec) * 1000000
                                           + (t_ransac_end.tv_usec - t_ransac_start.tv_usec)) / 1000;
                
                if (countNonZero(H) == 0) {
                    skippedDueToZeroH++;
                    continue;
                }

                Rect bRect1 = boundingRect(task.points1);

                pthread_mutex_lock(&mutex);
                rankedResultsOut.push_back(SearchResult(i_maxVal, i_imageId, bRect1));
                pthread_mutex_unlock(&mutex);
                
                successfulRansacs++;
                
                cout << "[RANSACThread] RANSAC for image " << i_imageId 
                     << " took " << ransac_time << " ms with " 
                     << task.points1.size() << " points, max val: " << i_maxVal << endl;
            }
            else {
                skippedDueToFewPoints++;
            }
        }
        else {
            skippedDueToLowValue++;
        }
    }
    
    gettimeofday(&t_end, NULL);
    unsigned long total_time = ((t_end.tv_sec - t_start.tv_sec) * 1000000
                              + (t_end.tv_usec - t_start.tv_usec)) / 1000;
    
    cout << "[RANSACThread] Processed " << imageIds.size() << " images in " << total_time << " ms" << endl;
    cout << "[RANSACThread] RANSAC attempts: " << ransacAttempts 
         << ", successful: " << successfulRansacs 
         << ", skipped (low val): " << skippedDueToLowValue
         << ", skipped (few points): " << skippedDueToFewPoints
         << ", skipped (zero H): " << skippedDueToZeroH << endl;
    
    return NULL;
}


/**
 * @brief Rerank images using a vector of sorted results.
 * @param imagesReqHits the hits of the request image.
 * @param indexHits the hits of the index.
 * @param sortedResults the sorted vector of results (weight, imageId).
 * @param i_nbResults the number of results to rerank.
 * @return A vector of reranked search results.
 */
vector<SearchResult> ImageReranker::rerank(unordered_map<u_int32_t, list<Hit> > &imagesReqHits,
                                         unordered_map<u_int32_t, const vector<Hit>* > &indexHits,
                                         const vector<pair<float, u_int32_t>> &sortedResults,
                                         unsigned i_nbResults)
{
    struct timeval t_start, t_extract;
    gettimeofday(&t_start, NULL);
    
    unordered_set<u_int32_t> firstImageIds;

    // Extract the first i_nbResults ranked images from the vector.
    getFirstImageIds(sortedResults, i_nbResults, firstImageIds);
    
    gettimeofday(&t_extract, NULL);
    cout << "[ImageReranker] Extracted " << firstImageIds.size() << " top images from vector in " 
         << ((t_extract.tv_sec - t_start.tv_sec) * 1000000 + (t_extract.tv_usec - t_start.tv_usec)) / 1000 
         << " ms" << endl;
         
    // Continue with the common reranking logic
    return rerankCommon(imagesReqHits, indexHits, firstImageIds);
}

/**
 * @brief Return the first ids of ranked images from a sorted vector.
 * @param sortedResults the sorted vector of results (weight, imageId).
 * @param i_nbResults the number of images to return.
 * @param firstImageIds a set to return the image ids.
 */
void ImageReranker::getFirstImageIds(const vector<pair<float, u_int32_t>> &sortedResults,
                                    unsigned i_nbResults, unordered_set<u_int32_t> &firstImageIds)
{
    unsigned i_res = 0;
    for (const auto& result : sortedResults)
    {
        if (i_res >= i_nbResults)
            break;
        
        firstImageIds.insert(result.second); // Insert the image ID
        i_res++;
    }
}

/**
 * @brief Common reranking implementation.
 * @param imagesReqHits the hits of the request image.
 * @param indexHits the hits of the index.
 * @param firstImageIds the set of image IDs to rerank.
 * @return A vector of reranked search results.
 */
vector<SearchResult> ImageReranker::rerankCommon(unordered_map<u_int32_t, list<Hit> > &imagesReqHits,
                                               unordered_map<u_int32_t, const vector<Hit>* > &indexHits,
                                               unordered_set<u_int32_t> &firstImageIds)
{
    struct timeval t_start, t_extract, t_histogram, t_threads, t_end;
    gettimeofday(&t_start, NULL);
    t_extract = t_start; // For timing consistency with old code

    unordered_map<u_int32_t, RANSACTask> imgTasks;

    // Compute the histograms.
    unordered_map<u_int32_t, Histogram> histograms; // key: the image id, value: the corresponding histogram.
    
    unsigned totalMatches = 0;
    unsigned totalHistogramEntries = 0;
    unsigned totalPointPairs = 0;

    for (unordered_map<u_int32_t, list<Hit> >::const_iterator it = imagesReqHits.begin();
         it != imagesReqHits.end(); ++it)
    {
        // Try to match all the visual words of the request image.
        const unsigned i_wordId = it->first;
        const list<Hit> &hits = it->second;

        assert(hits.size() == 1);

        // If there is several hits for the same word in the image...
        const u_int16_t i_angle1 = hits.front().i_angle;
        const Point2f point1(hits.front().x, hits.front().y);
        const vector<Hit> *hitIndex = indexHits[i_wordId];
        
        if (!hitIndex) {
            continue;
        }
        
        unsigned matchesForThisWord = 0;

        for (unsigned i = 0; i < hitIndex->size(); ++i)
        {
            const u_int32_t i_imageId = (*hitIndex)[i].i_imageId;
            // Test if the image belongs to the image to rerank.
            if (firstImageIds.find(i_imageId) != firstImageIds.end())
            {
                matchesForThisWord++;
                totalMatches++;
                
                const u_int16_t i_angle2 = (*hitIndex)[i].i_angle;
                float f_diff = angleDiff(i_angle1, i_angle2);
                unsigned bin = (f_diff - DIFF_MIN) / 360 * HISTOGRAM_NB_BINS;
                assert(bin < HISTOGRAM_NB_BINS);

                Histogram &histogram = histograms[i_imageId];
                histogram.bins[bin]++;
                histogram.i_total++;
                totalHistogramEntries++;

                const Point2f point2((*hitIndex)[i].x, (*hitIndex)[i].y);
                RANSACTask &imgTask = imgTasks[i_imageId];

                imgTask.points1.push_back(point1);
                imgTask.points2.push_back(point2);
                totalPointPairs++;
            }
        }
        
        if (matchesForThisWord > 0) {
            cout << "[ImageReranker] Word " << i_wordId << " matched " << matchesForThisWord << " images" << endl;
        }
    }
    
    gettimeofday(&t_histogram, NULL);
    cout << "[ImageReranker] Built histograms in " 
         << ((t_histogram.tv_sec - t_extract.tv_sec) * 1000000 + (t_histogram.tv_usec - t_extract.tv_usec)) / 1000 
         << " ms" << endl;
    cout << "[ImageReranker] Total matches: " << totalMatches 
         << ", histogram entries: " << totalHistogramEntries 
         << ", point pairs: " << totalPointPairs << endl;
    cout << "[ImageReranker] Images with histograms: " << histograms.size() 
         << ", images with point pairs: " << imgTasks.size() << endl;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    // Create a vector to store the results
    vector<SearchResult> rankedResults;
    rankedResults.reserve(histograms.size()); // Reserve space for efficiency

    #define NB_RANSAC_THREAD 4
    RANSACThread *threads[NB_RANSAC_THREAD];

    for (unsigned i = 0; i < NB_RANSAC_THREAD; ++i)
        threads[i] = new RANSACThread(mutex, imgTasks, rankedResults);

    // Rank the images according to their histogram.
    unsigned i = 0;
    for (unordered_map<unsigned, Histogram>::iterator it = histograms.begin();
         it != histograms.end(); ++it, ++i)
    {
        unsigned i_imageId = it->first;
        Histogram histogram = it->second;
        threads[i % NB_RANSAC_THREAD]->imageIds.push_back(i_imageId);
        threads[i % NB_RANSAC_THREAD]->histograms.push_back(histogram);
    }

    // Compute
    gettimeofday(&t_threads, NULL);
    cout << "[ImageReranker] Thread setup time: " 
         << ((t_threads.tv_sec - t_histogram.tv_sec) * 1000000 + (t_threads.tv_usec - t_histogram.tv_usec)) / 1000 
         << " ms" << endl;
         
    for (unsigned i = 0; i < NB_RANSAC_THREAD; ++i)
        threads[i]->start();
    for (unsigned i = 0; i < NB_RANSAC_THREAD; ++i)
    {
        threads[i]->join();
        delete threads[i];
    }

    pthread_mutex_destroy(&mutex);
    
    gettimeofday(&t_end, NULL);
    cout << "[ImageReranker] RANSAC threads total time: " 
         << ((t_end.tv_sec - t_threads.tv_sec) * 1000000 + (t_end.tv_usec - t_threads.tv_usec)) / 1000 
         << " ms" << endl;
    cout << "[ImageReranker] Total reranking time: " 
         << ((t_end.tv_sec - t_start.tv_sec) * 1000000 + (t_end.tv_usec - t_start.tv_usec)) / 1000 
         << " ms" << endl;
    
    // Sort the results by weight in descending order
    sort(rankedResults.begin(), rankedResults.end(), 
         [](const SearchResult& a, const SearchResult& b) {
             return a.f_weight > b.f_weight;
         });
    
    return rankedResults;
}


class Pos {
public:
    Pos(int x, int y) : x(x), y(y) {}

    inline bool operator< (const Pos &rhs) const {
        if (x != rhs.x)
            return x < rhs.x;
        else
            return y < rhs.y;
    }

private:
    int x, y;
};


float ImageReranker::angleDiff(unsigned i_angle1, unsigned i_angle2)
{
    // Convert the angle in the [-180, 180] range.
    float i1 = (float)i_angle1 * 360 / (1 << 16);
    float i2 = (float)i_angle2 * 360 / (1 << 16);

    i1 = i1 <= 180 ? i1 : i1 - 360;
    i2 = i2 <= 180 ? i2 : i2 - 360;

    // Compute the difference between the two angles.
    float diff = i1 - i2;
    if (diff < DIFF_MIN)
        diff += 360;
    else if (diff >= 360 + DIFF_MIN)
        diff -= 360;

    assert(diff >= DIFF_MIN);
    assert(diff < 360 + DIFF_MIN);

    return diff;
}
