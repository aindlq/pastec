# Pastec Performance Optimization Plan

This document outlines a comprehensive plan for optimizing the Pastec image search system. Based on performance logs and code analysis, we identify current bottlenecks and propose both incremental improvements and a complete redesign approach that takes advantage of high-memory environments (256GB RAM).

## Table of Contents

1. [Current Performance Analysis](#current-performance-analysis)
2. [Key Bottlenecks](#key-bottlenecks)
3. [Current Implementation Analysis](#current-implementation-analysis)
4. [Incremental Improvements](#incremental-improvements)
5. [Complete Redesign Approach](#complete-redesign-approach)
6. [SIMD Optimization Opportunities](#simd-optimization-opportunities)
7. [Implementation Roadmap](#implementation-roadmap)

## Current Performance Analysis

The performance log shows that a typical search operation takes approximately 30-32 seconds, with the following breakdown:

```
[TIMING] http_parse_json                         : 0 ms
[TIMING] http_download_image                     : 1277 ms
[TIMING] image_loading                           : 1 ms
[TIMING] orb_detection                           : 41 ms
[TIMING] word_search                             : 135 ms
[TIMING] getImagesWithVisualWords_prep           : 0 ms
[TIMING] getImagesWithVisualWords_total          : 505 ms
[TIMING] ranking_thread_compute                  : 28406 ms
[TIMING] ranking_reduce                          : 487 ms
[TIMING] reranking                               : 742 ms
[TIMING] processSimilar_total                    : 30594 ms
[TIMING] searchImage_total                       : 30780 ms
[TIMING] http_search_total                       : 32058 ms
```

Memory access statistics reveal significant data movement:

```
===== MEMORY ACCESS STATISTICS =====
Data Structure                Read Count     Read Bytes     Write Count    Write Bytes    
------------------------------------------------------------------------------------------
weights                       0              0              167271456      663714124      
nbWords                       186529527      721384916      0              0              
indexHitsForReq               0              0              2232           2623845528     
indexHits                     4464           5247691056     0              0              
nbOccurences                  4848           19392          0              0              
```

## Key Bottlenecks

1. **Ranking Phase (28.4s)**: The most significant bottleneck, consuming over 90% of the total search time
   - Multiple threads processing hits across millions of images
   - 186 million reads from `nbWords` (721MB)
   - 167 million writes to `weights` (663MB)

2. **Memory Access Patterns**:
   - Large data copies in `getImagesWithVisualWords` (2.6GB)
   - Excessive reads from `indexHits` (5GB)
   - Redundant lookups in `countTotalNbWord`

3. **Thread Synchronization**:
   - Lock contention in index access (unnecessary since the index is read-only during queries)
   - Inefficient work distribution among threads

4. **Algorithm Inefficiencies**:
   - Full scan of all hits without early termination
   - Inefficient reduction of weights from multiple threads

## Current Implementation Analysis

### Image Search Pipeline

The current search pipeline in `ORBSearcher::searchImage` follows these steps:

1. **Image Loading and Feature Extraction**:
   ```cpp
   // src/orb/orbsearcher.cpp:118-126
   TimingStats::startTimer("image_loading");
   Mat img;
   u_int32_t i_ret = ImageLoader::loadImage(request.imageData.size(),
                                          request.imageData.data(), img);
   TimingStats::endTimer("image_loading");

   TimingStats::startTimer("orb_detection");
   vector<KeyPoint> keypoints;
   Mat descriptors;
   orb->detectAndCompute(img, noArray(), keypoints, descriptors);
   TimingStats::endTimer("orb_detection");
   ```

2. **Visual Word Search**:
   ```cpp
   // src/orb/orbsearcher.cpp:139-177
   TimingStats::startTimer("word_search");
   std::unordered_map<u_int32_t, list<Hit> > imageReqHits;
   
   for (unsigned i = 0; i < keypoints.size(); ++i) {
       // KNN search to find matching visual words
       wordIndex->knnSearch(descriptors.row(i), indices, dists, NB_NEIGHBORS);
       
       // Filter words based on occurrence count
       if (index->getWordNbOccurences(i_wordId) > i_maxNbOccurences)
           continue;
           
       // Add hit to request hits
       imageReqHits[i_wordId].push_back(hit);
   }
   TimingStats::endTimer("word_search");
   ```

3. **Hit Collection**:
   ```cpp
   // src/orb/orbindex.cpp:77-121
   TimingStats::startTimer("getImagesWithVisualWords_total");
   
   // Pre-allocate memory for the result
   indexHitsForReq.reserve(imagesReqHits.size());
   
   // Copy hits for each word
   for (unordered_map<u_int32_t, list<Hit> >::const_iterator it = imagesReqHits.begin();
        it != imagesReqHits.end(); ++it) {
       const unsigned i_wordId = it->first;
       
       // Lock only when accessing the specific word
       pthread_rwlock_rdlock(&rwLock);
       const vector<Hit>& hits = indexHits[i_wordId];
       pthread_rwlock_unlock(&rwLock);
       
       // Deep copy of hits
       vector<Hit>& destHits = indexHitsForReq[i_wordId];
       destHits.reserve(hits.size());
       destHits = hits;  // This is a deep copy
   }
   
   TimingStats::endTimer("getImagesWithVisualWords_total");
   ```

4. **Ranking** (the main bottleneck):
   ```cpp
   // src/orb/orbsearcher.cpp:253-282
   // Map the ranking to threads
   unsigned i_wordsPerThread = indexHits.size() / NB_RANKING_THREAD + 1;
   RankingThread *threads[NB_RANKING_THREAD];
   
   // Distribute words to threads
   for (unsigned i = 0; i < NB_RANKING_THREAD; ++i) {
       threads[i] = new RankingThread(index, i_nbTotalIndexedImages, indexHits);
       // Add words to this thread
       for (; it != indexHits.end() && i_nbWords < i_wordsPerThread; ++it, ++i_nbWords)
           threads[i]->addWord(it->first);
   }
   
   // Compute in parallel
   for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
       threads[i]->start();
   for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
       threads[i]->join();
   ```

5. **Ranking Thread Implementation**:
   ```cpp
   // src/orb/orbsearcher.cpp:63-104
   void *run() {
       TimingStats::startTimer("RankingThread_run");
       weights.rehash(wordIds.size());
       
       for (deque<u_int32_t>::const_iterator it = wordIds.begin();
           it != wordIds.end(); ++it) {
           const vector<Hit> &hits = indexHits[*it];
           
           const float f_weight = log((float)i_nbTotalIndexedImages / hits.size());

           for (vector<Hit>::const_iterator it2 = hits.begin();
                it2 != hits.end(); ++it2) {
               // TF-IDF calculation
               TimingStats::startTimer("countTotalNbWord");
               unsigned i_totalNbWords = index->countTotalNbWord(it2->i_imageId);
               TimingStats::endTimer("countTotalNbWord", false);
               
               weights[it2->i_imageId] += f_weight / i_totalNbWords;
           }
       }
       
       TimingStats::endTimer("RankingThread_run");
       return NULL;
   }
   ```

6. **Weight Reduction**:
   ```cpp
   // src/orb/orbsearcher.cpp:293-301
   TimingStats::startTimer("ranking_reduce");
   std::unordered_map<u_int32_t, float> weights;
   weights.rehash(i_nbTotalIndexedImages);
   
   for (unsigned i = 0; i < NB_RANKING_THREAD; ++i) {
       for (std::unordered_map<u_int32_t, float>::const_iterator it = threads[i]->weights.begin();
           it != threads[i]->weights.end(); ++it)
           weights[it->first] += it->second;
   }
   TimingStats::endTimer("ranking_reduce");
   ```

7. **Reranking and Result Return**:
   ```cpp
   // src/orb/orbsearcher.cpp:318-324
   TimingStats::startTimer("reranking");
   priority_queue<SearchResult> rerankedResults;
   reranker.rerank(imageReqHits, indexHits,
                   rankedResults, rerankedResults, 300);
   TimingStats::endTimer("reranking");
   
   returnResults(rerankedResults, request, 100);
   ```

### Key Inefficiencies

1. **Excessive Memory Copies**:
   - In `getImagesWithVisualWords`, the entire hit vector is copied for each word
   - This creates a duplicate of the index in memory

2. **Redundant Lookups**:
   - `countTotalNbWord` is called 186 million times, each requiring a lock and hash table lookup
   - This function is extremely simple but called excessively:
   ```cpp
   // src/orb/orbindex.cpp:127-134
   unsigned ORBIndex::countTotalNbWord(unsigned i_imageId) {
       // Track memory access for reading nbWords
       MemoryAccessTracker::recordAccess("nbWords", sizeof(unsigned));
       
       // Lookup in the unordered_map
       unsigned i_ret = nbWords[i_imageId];
       return i_ret;
   }
   ```

3. **Unnecessary Locks**:
   - The current implementation uses locks even though the index is read-only during queries
   - Multiple read locks are acquired and released for each word lookup
   - The `readLock()` and `unlock()` pattern is used extensively but is unnecessary

4. **Inefficient Thread Work Distribution**:
   - Work is divided by word count, not by actual computational load
   - This leads to thread imbalance as shown in the timing logs:
   ```
   [TIMING] RankingThread_run                       : 22631 ms
   [TIMING] RankingThread_run                       : 24697 ms
   [TIMING] RankingThread_run                       : 25743 ms
   [TIMING] RankingThread_run                       : 26124 ms
   [TIMING] RankingThread_run                       : 26711 ms
   [TIMING] RankingThread_run                       : 27354 ms
   [TIMING] RankingThread_run                       : 27911 ms
   [TIMING] RankingThread_run                       : 28284 ms
   [TIMING] RankingThread_run                       : 28393 ms
   [TIMING] RankingThread_run                       : 28404 ms
   ```

## Incremental Improvements

These improvements can be implemented without a complete redesign:

### 1. Cache Word Counts

**Current Implementation**:
```cpp
// src/orb/orbsearcher.cpp:85-89
TimingStats::startTimer("countTotalNbWord");
unsigned i_totalNbWords = index->countTotalNbWord(it2->i_imageId);
TimingStats::endTimer("countTotalNbWord", false);

weights[it2->i_imageId] += f_weight / i_totalNbWords;
```

**Proposed Improvement**:
```cpp
// Cache word counts in a thread-local unordered_map
std::unordered_map<u_int32_t, unsigned> wordCountCache;

// In the ranking thread
unsigned getWordCount(u_int32_t imageId) {
    auto it = wordCountCache.find(imageId);
    if (it != wordCountCache.end()) {
        return it->second;
    }
    
    // No locks needed since the index is read-only during queries
    unsigned count = index->nbWords[imageId];
    wordCountCache[imageId] = count;
    return count;
}

// Then use this in the ranking loop
unsigned i_totalNbWords = getWordCount(it2->i_imageId);
weights[it2->i_imageId] += f_weight / i_totalNbWords;
```

**Expected Improvement**: Reduces 186 million lookups to approximately 1.7 million (the number of unique images).

### 2. Eliminate Unnecessary Locks and Copies

**Current Implementation**:
```cpp
// src/orb/orbindex.cpp:102-110
// Lock only when accessing the specific word
pthread_rwlock_rdlock(&rwLock);
const vector<Hit>& hits = indexHits[i_wordId];
pthread_rwlock_unlock(&rwLock);

// Deep copy of hits
vector<Hit>& destHits = indexHitsForReq[i_wordId];
destHits.reserve(hits.size());
destHits = hits;  // This is a deep copy
```

**Proposed Improvement**:
```cpp
// No locks needed since the index is read-only during queries
const vector<Hit>& hits = indexHits[i_wordId];

// Use references instead of deep copies
indexHitsForReq[i_wordId] = &hits;  // Store reference to original data
```

**Expected Improvement**: Eliminates 2.6GB of memory copies and removes lock overhead, reducing `getImagesWithVisualWords_total` time from 505ms to under 50ms.

### 3. Improve Thread Work Distribution

**Current Implementation**:
```cpp
// src/orb/orbsearcher.cpp:253-267
unsigned i_wordsPerThread = indexHits.size() / NB_RANKING_THREAD + 1;
// ...
for (unsigned i = 0; i < NB_RANKING_THREAD; ++i) {
    threads[i] = new RankingThread(index, i_nbTotalIndexedImages, indexHits);

    unsigned i_nbWords = 0;
    for (; it != indexHits.end() && i_nbWords < i_wordsPerThread; ++it, ++i_nbWords)
        threads[i]->addWord(it->first);
}
```

**Proposed Improvement**:
```cpp
// Distribute work based on estimated computational load
std::vector<std::pair<u_int32_t, size_t>> wordLoads;
for (auto it = indexHits.begin(); it != indexHits.end(); ++it) {
    wordLoads.push_back({it->first, it->second.size()});
}

// Sort by computational load (hit count)
std::sort(wordLoads.begin(), wordLoads.end(), 
    [](const auto& a, const auto& b) { return a.second > b.second; });

// Distribute words to threads using a greedy algorithm
std::vector<size_t> threadLoads(NB_RANKING_THREAD, 0);
std::vector<std::vector<u_int32_t>> threadWords(NB_RANKING_THREAD);

for (const auto& wordLoad : wordLoads) {
    // Find thread with minimum current load
    int minThread = 0;
    for (int i = 1; i < NB_RANKING_THREAD; i++) {
        if (threadLoads[i] < threadLoads[minThread]) {
            minThread = i;
        }
    }
    
    // Assign word to this thread
    threadWords[minThread].push_back(wordLoad.first);
    threadLoads[minThread] += wordLoad.second;
}

// Create and start threads
for (unsigned i = 0; i < NB_RANKING_THREAD; ++i) {
    threads[i] = new RankingThread(index, i_nbTotalIndexedImages, indexHits);
    for (u_int32_t wordId : threadWords[i]) {
        threads[i]->addWord(wordId);
    }
}
```

**Expected Improvement**: Better load balancing reduces the maximum thread runtime, potentially improving ranking time by 15-20%.

### 4. Batch Processing for TF-IDF Calculation

**Current Implementation**:
```cpp
// src/orb/orbsearcher.cpp:85-89
for (vector<Hit>::const_iterator it2 = hits.begin(); it2 != hits.end(); ++it2) {
    TimingStats::startTimer("countTotalNbWord");
    unsigned i_totalNbWords = index->countTotalNbWord(it2->i_imageId);
    TimingStats::endTimer("countTotalNbWord", false);
    
    weights[it2->i_imageId] += f_weight / i_totalNbWords;
}
```

**Proposed Improvement**:
```cpp
// Process hits in batches to improve cache locality
const size_t BATCH_SIZE = 64;  // Adjust based on cache line size
std::vector<u_int32_t> imageIds(BATCH_SIZE);
std::vector<unsigned> wordCounts(BATCH_SIZE);
std::vector<float> weightIncrements(BATCH_SIZE);

for (size_t i = 0; i < hits.size(); i += BATCH_SIZE) {
    // Determine actual batch size for this iteration
    size_t batchEnd = std::min(i + BATCH_SIZE, hits.size());
    size_t actualBatchSize = batchEnd - i;
    
    // Collect image IDs for this batch
    for (size_t j = 0; j < actualBatchSize; j++) {
        imageIds[j] = hits[i + j].i_imageId;
    }
    
    // Batch lookup word counts (no locks needed)
    for (size_t j = 0; j < actualBatchSize; j++) {
        wordCounts[j] = getWordCount(imageIds[j]);
    }
    
    // Calculate weight increments
    for (size_t j = 0; j < actualBatchSize; j++) {
        weightIncrements[j] = f_weight / wordCounts[j];
    }
    
    // Update weights
    for (size_t j = 0; j < actualBatchSize; j++) {
        weights[imageIds[j]] += weightIncrements[j];
    }
}
```

**Expected Improvement**: Improves cache locality and reduces memory access patterns, potentially improving ranking performance by 20-30%.

## Complete Redesign Approach

Given the availability of 256GB RAM and the assumption that the index is built in an ETL pipeline and then only queried (read-only operations), a complete redesign can prioritize performance over memory efficiency:

> **Important Assumption**: The index is built in a separate ETL process and is only read (not modified) during search operations. This eliminates the need for locks and thread synchronization during queries, significantly simplifying the design and improving performance.

### 1. Memory-First Index Design

#### Columnar Storage Format

**Current Implementation**:
```cpp
// Current index structure (src/orb/orbindex.h)
vector<Hit> indexHits[NB_VISUAL_WORDS];  // Array of vectors
```

**Proposed Redesign**:
```cpp
// Columnar storage for better cache locality and SIMD processing
struct ColumnStore {
    // Each array is contiguous in memory
    std::vector<uint32_t> imageIds;  // All image IDs in one array
    std::vector<uint16_t> angles;    // All angles in one array
    std::vector<uint16_t> x_coords;  // All x coordinates in one array
    std::vector<uint16_t> y_coords;  // All y coordinates in one array
    
    // Index mapping from wordId to range in the arrays
    std::vector<std::pair<size_t, size_t>> wordRanges;
};

// Replace indexHits with this structure
ColumnStore columnStore;
```

#### Pre-computed Word Counts

**Current Implementation**:
```cpp
// src/orb/orbindex.cpp:127-134
unsigned ORBIndex::countTotalNbWord(unsigned i_imageId) {
    // Lookup in the unordered_map
    unsigned i_ret = nbWords[i_imageId];
    return i_ret;
}
```

**Proposed Redesign**:
```cpp
// Pre-compute and store word counts in a flat array for direct access
class OptimizedIndex {
public:
    OptimizedIndex(const ORBIndex& originalIndex) {
        // Get all image IDs
        std::vector<uint32_t> imageIds;
        originalIndex.getImageIds(imageIds);
        
        // Find maximum image ID to determine array size
        uint32_t maxImageId = 0;
        for (uint32_t id : imageIds) {
            maxImageId = std::max(maxImageId, id);
        }
        
        // Allocate array with size maxImageId + 1
        wordCounts.resize(maxImageId + 1, 0);
        
        // Fill array with word counts
        for (uint32_t id : imageIds) {
            wordCounts[id] = originalIndex.nbWords.at(id);
        }
    }
    
    // Direct array access - no locks, no hash table lookup
    inline unsigned getWordCount(uint32_t imageId) const {
        return wordCounts[imageId];
    }
    
private:
    std::vector<unsigned> wordCounts;
};
```

#### Thread-Local Index Copies

**Current Implementation**:
```cpp
// src/orb/orbsearcher.cpp:253-267
// All threads share the same index with locks
RankingThread *threads[NB_RANKING_THREAD];
for (unsigned i = 0; i < NB_RANKING_THREAD; ++i) {
    threads[i] = new RankingThread(index, i_nbTotalIndexedImages, indexHits);
    // ...
}
```

**Proposed Redesign**:
```cpp
// Since the index is read-only during queries, we can use direct access
// without locks. For maximum performance, we can create thread-local
// copies of frequently accessed data.

struct ThreadLocalData {
    // Thread-local memory pool for allocations
    MemoryPool pool;
    
    // Thread-local result accumulation
    std::unordered_map<uint32_t, float> localWeights;
    
    // Pre-allocated buffers for batch processing
    std::vector<uint32_t> imageIdBatch;
    std::vector<float> weightBatch;
};

// Initialize thread-local data
std::vector<ThreadLocalData> threadData(NUM_THREADS);
for (int i = 0; i < NUM_THREADS; i++) {
    threadData[i].imageIdBatch.reserve(BATCH_SIZE);
    threadData[i].weightBatch.reserve(BATCH_SIZE);
}
```

### 2. Lock-Free Parallel Processing with Immutable Index

#### Immutable Index with No Locks

**Current Implementation**:
```cpp
// src/orb/orbindex.cpp:77-121
// Uses locks for thread safety
pthread_rwlock_rdlock(&rwLock);
const vector<Hit>& hits = indexHits[i_wordId];
pthread_rwlock_unlock(&rwLock);
```

**Proposed Redesign**:
```cpp
// No locks needed since the index is immutable during queries
const auto& hits = indexHits[i_wordId];  // Direct access, no locks

// Only need thread-safe structures for accumulating results
tbb::concurrent_unordered_map<u_int32_t, float> weights;
weights[imageId] += value;  // Thread-safe update for results
```

This approach completely eliminates lock overhead during query operations, which is a significant performance improvement. Since the index is built in a separate ETL process and not modified during search, we can safely access it without locks.

#### Work Stealing Scheduler

**Current Implementation**:
```cpp
// Static work division (src/orb/orbsearcher.cpp:253-267)
unsigned i_wordsPerThread = indexHits.size() / NB_RANKING_THREAD + 1;
// ...
```

**Proposed Redesign**:
```cpp
// Work stealing scheduler
class WorkStealingScheduler {
public:
    WorkStealingScheduler(const std::vector<u_int32_t>& wordIds, size_t numThreads) 
        : numThreads_(numThreads) {
        // Create task queues for each thread
        queues_.resize(numThreads_);
        
        // Distribute initial tasks
        distributeInitialTasks(wordIds);
    }
    
    // Get next task for a thread
    std::optional<u_int32_t> getNextTask(size_t threadId) {
        // Try to get task from own queue
        if (!queues_[threadId].empty()) {
            u_int32_t task = queues_[threadId].back();
            queues_[threadId].pop_back();
            return task;
        }
        
        // Try to steal from other queues
        for (size_t i = 0; i < numThreads_; i++) {
            if (i == threadId) continue;
            
            // Try to steal from the front (opposite end)
            std::lock_guard<std::mutex> lock(locks_[i]);
            if (!queues_[i].empty()) {
                u_int32_t task = queues_[i].front();
                queues_[i].pop_front();
                return task;
            }
        }
        
        // No tasks available
        return std::nullopt;
    }
    
private:
    void distributeInitialTasks(const std::vector<u_int32_t>& wordIds) {
        // Sort words by estimated work
        // ...
        
        // Distribute in round-robin fashion
        for (size_t i = 0; i < wordIds.size(); i++) {
            queues_[i % numThreads_].push_back(wordIds[i]);
        }
    }
    
    size_t numThreads_;
    std::vector<std::deque<u_int32_t>> queues_;
    std::vector<std::mutex> locks_;
};
```

### 3. SIMD-Accelerated Ranking

#### Vectorized TF-IDF Calculation

**Current Implementation**:
```cpp
// src/orb/orbsearcher.cpp:85-89
for (vector<Hit>::const_iterator it2 = hits.begin(); it2 != hits.end(); ++it2) {
    unsigned i_totalNbWords = index->countTotalNbWord(it2->i_imageId);
    weights[it2->i_imageId] += f_weight / i_totalNbWords;
}
```

**Proposed Redesign**:
```cpp
// SIMD-accelerated TF-IDF calculation
#include <immintrin.h>  // For AVX intrinsics

void processHitsBatch(const uint32_t* imageIds, const unsigned* wordCounts, 
                     float f_weight, float* weightUpdates, size_t count) {
    // Process in chunks of 8 (AVX2) or 16 (AVX-512)
    const size_t simdWidth = 8;  // For AVX2
    
    // Broadcast f_weight to all SIMD lanes
    __m256 weightVec = _mm256_set1_ps(f_weight);
    
    for (size_t i = 0; i < count; i += simdWidth) {
        // Handle remaining elements if not multiple of SIMD width
        size_t remainingElements = std::min(simdWidth, count - i);
        
        if (remainingElements == simdWidth) {
            // Load word counts
            float countsFloat[simdWidth];
            for (size_t j = 0; j < simdWidth; j++) {
                countsFloat[j] = static_cast<float>(wordCounts[i + j]);
            }
            __m256 countsVec = _mm256_loadu_ps(countsFloat);
            
            // Calculate f_weight / wordCount
            __m256 resultVec = _mm256_div_ps(weightVec, countsVec);
            
            // Store results
            float results[simdWidth];
            _mm256_storeu_ps(results, resultVec);
            
            // Update weight map
            for (size_t j = 0; j < simdWidth; j++) {
                weightUpdates[i + j] = results[j];
            }
        } else {
            // Handle remaining elements scalar
            for (size_t j = 0; j < remainingElements; j++) {
                weightUpdates[i + j] = f_weight / wordCounts[i + j];
            }
        }
    }
}
```

#### Vectorized Weight Accumulation

**Current Implementation**:
```cpp
// src/orb/orbsearcher.cpp:293-301
for (unsigned i = 0; i < NB_RANKING_THREAD; ++i) {
    for (std::unordered_map<u_int32_t, float>::const_iterator it = threads[i]->weights.begin();
        it != threads[i]->weights.end(); ++it)
        weights[it->first] += it->second;
}
```

**Proposed Redesign**:
```cpp
// SIMD-accelerated weight accumulation
void accumulateWeights(float* destWeights, const float* srcWeights, size_t count) {
    const size_t simdWidth = 8;  // For AVX2
    
    for (size_t i = 0; i < count; i += simdWidth) {
        size_t remainingElements = std::min(simdWidth, count - i);
        
        if (remainingElements == simdWidth) {
            // Load current weights
            __m256 destVec = _mm256_loadu_ps(&destWeights[i]);
            __m256 srcVec = _mm256_loadu_ps(&srcWeights[i]);
            
            // Add weights
            __m256 resultVec = _mm256_add_ps(destVec, srcVec);
            
            // Store results
            _mm256_storeu_ps(&destWeights[i], resultVec);
        } else {
            // Handle remaining elements scalar
            for (size_t j = 0; j < remainingElements; j++) {
                destWeights[i + j] += srcWeights[i + j];
            }
        }
    }
}
```

## SIMD Optimization Opportunities

SIMD instructions can be applied throughout the pipeline:

### 1. Feature Extraction and Matching

**Current Implementation**:
```cpp
// src/orb/orbwordindex.cpp:41-47
// Uses OpenCV's built-in KNN search
void knnSearch(const Mat& query, vector<int>& indices,
              vector<int>& dists, int knn) {
    cvflann::KNNResultSet<int> m_indices(knn);
    m_indices.init(indices.data(), dists.data());
    kdIndex->findNeighbors(m_indices, (unsigned char*)query.ptr<unsigned char>(0),
                          cvflann::SearchParams(2000));
}
```

**Proposed SIMD Implementation**:
```cpp
// Batch KNN search with SIMD acceleration
void batchKnnSearch(const Mat& queries, std::vector<std::vector<int>>& indices,
                   std::vector<std::vector<int>>& dists, int knn, int batchSize) {
    // Process multiple queries in parallel
    #pragma omp parallel for
    for (int b = 0; b < queries.rows; b += batchSize) {
        int actualBatchSize = std::min(batchSize, queries.rows - b);
        
        // Pre-compute distances using SIMD
        std::vector<std::vector<std::pair<float, int>>> distances(actualBatchSize);
        
        // For each word in the vocabulary
        for (int wordIdx = 0; wordIdx < words->rows; wordIdx++) {
            // Compute Hamming distances for the batch using SIMD
            uint32_t batchDistances[actualBatchSize];
            computeBatchHammingDistances(
                (uint8_t*)queries.ptr(b), 
                (uint8_t*)words->ptr(wordIdx),
                queries.cols, actualBatchSize, batchDistances);
            
            // Store distances
            for (int i = 0; i < actualBatchSize; i++) {
                distances[i].push_back({batchDistances[i], wordIdx});
            }
        }
        
        // Sort and extract top-k for each query
        for (int i = 0; i < actualBatchSize; i++) {
            std::partial_sort(distances[i].begin(), 
                             distances[i].begin() + knn,
                             distances[i].end());
            
            // Extract results
            for (int k = 0; k < knn; k++) {
                indices[b + i].push_back(distances[i][k].second);
                dists[b + i].push_back(distances[i][k].first);
            }
        }
    }
}

// SIMD-accelerated Hamming distance computation for batches
void computeBatchHammingDistances(const uint8_t* queries, const uint8_t* word,
                                 int dims, int batchSize, uint32_t* distances) {
    // Use AVX2 for 256-bit SIMD
    for (int i = 0; i < batchSize; i++) {
        const uint8_t* query = queries + i * dims;
        distances[i] = hammingDistanceAVX2(query, word, dims);
    }
}

// AVX2 implementation of Hamming distance
uint32_t hammingDistanceAVX2(const uint8_t* a, const uint8_t* b, int size) {
    uint32_t distance = 0;
    int i = 0;
    
    // Process 32 bytes (256 bits) at a time using AVX2
    for (; i + 32 <= size; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i vxor = _mm256_xor_si256(va, vb);
        
        // Count bits using popcnt
        uint8_t bytes[32];
        _mm256_storeu_si256((__m256i*)bytes, vxor);
        
        for (int j = 0; j < 32; j++) {
            distance += _mm_popcnt_u32(bytes[j]);
        }
    }
    
    // Handle remaining bytes
    for (; i < size; i++) {
        distance += _mm_popcnt_u32(a[i] ^ b[i]);
    }
    
    return distance;
}
```

### 2. Hamming Distance Calculation

The current implementation already uses SimSIMD for Hamming distance calculation:

```cpp
// include/orb/orbwordindex.h:25-36
// Custom Hamming distance functor using SimSIMD
class SimSIMDHamming {
public:
    typedef unsigned char ElementType;
    typedef int ResultType;

    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 a, Iterator2 b, size_t size) const {
        simsimd_distance_t distance;
        simsimd_hamming_b8((simsimd_b8_t*)a, (simsimd_b8_t*)b, size, &distance);
        return (ResultType)distance;
    }
};
```

This can be extended to batch processing:

```cpp
// Batch Hamming distance calculation
void batchHammingDistance(const uint8_t* queries, const uint8_t* words,
                         int numQueries, int numWords, int dims, int* distances) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numQueries; i++) {
        for (int j = 0; j < numWords; j++) {
            simsimd_distance_t distance;
            simsimd_hamming_b8(
                (simsimd_b8_t*)(queries + i * dims),
                (simsimd_b8_t*)(words + j * dims),
                dims,
                &distance
            );
            distances[i * numWords + j] = distance;
        }
    }
}
```

### 3. Weight Calculation and Reduction

SIMD can significantly accelerate the weight calculation and reduction phases:

```cpp
// SIMD-accelerated weight calculation for multiple images
void calculateWeightsBatch(const uint32_t* imageIds, const unsigned* wordCounts,
                          float idf, float* weights, int count) {
    // Use AVX-512 if available (16 floats at once)
    #ifdef __AVX512F__
    const int simdWidth = 16;
    __m512 idfVec = _mm512_set1_ps(idf);
    
    for (int i = 0; i < count; i += simdWidth) {
        int remaining = std::min(simdWidth, count - i);
        
        if (remaining == simdWidth) {
            // Convert word counts to float
            float countsFloat[simdWidth];
            for (int j = 0; j < simdWidth; j++) {
                countsFloat[j] = static_cast<float>(wordCounts[i + j]);
            }
            
            // Load counts and calculate weights
            __m512 countsVec = _mm512_loadu_ps(countsFloat);
            __m512 resultVec = _mm512_div_ps(idfVec, countsVec);
            
            // Store results
            _mm512_storeu_ps(weights + i, resultVec);
        } else {
            // Handle remaining elements
            for (int j = 0; j < remaining; j++) {
                weights[i + j] = idf / wordCounts[i + j];
            }
        }
    }
    #else
    // Fallback to AVX2 (8 floats at once)
    const int simdWidth = 8;
    __m256 idfVec = _mm256_set1_ps(idf);
    
    for (int i = 0; i < count; i += simdWidth) {
        int remaining = std::min(simdWidth, count - i);
        
        if (remaining == simdWidth) {
            // Convert word counts to float
            float countsFloat[simdWidth];
            for (int j = 0; j < simdWidth; j++) {
                countsFloat[j] = static_cast<float>(wordCounts[i + j]);
            }
            
            // Load counts and calculate weights
            __m256 countsVec = _mm256_loadu_ps(countsFloat);
            __m256 resultVec = _mm256_div_ps(idfVec, countsVec);
            
            // Store results
            _mm256_storeu_ps(weights + i, resultVec);
        } else {
            // Handle remaining elements
            for (int j = 0; j < remaining; j++) {
                weights[i + j] = idf / wordCounts[i + j];
            }
        }
    }
    #endif
}
```

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

1. **Remove Unnecessary Locks**
   - Eliminate all locks in query operations since the index is read-only
   - Directly access index data structures without synchronization

2. **Implement Word Count Caching**
   - Add thread-local caching in RankingThread
   - Modify countTotalNbWord to use direct access without locks

3. **Optimize Hit Collection**
   - Replace deep copies with references to original data
   - Update all code that accesses indexHitsForReq

4. **Improve Thread Work Distribution**
   - Implement load-balanced work distribution
   - Test with different distribution strategies

### Phase 2: Algorithm Improvements (2-4 weeks)

1. **Batch Processing**
   - Implement batch processing for TF-IDF calculation
   - Add SIMD acceleration for weight calculation

2. **Early Termination**
   - Add threshold-based early termination
   - Implement approximate ranking for initial filtering

3. **Optimize Memory Access Patterns**
   - Reorganize data for better cache locality
   - Implement software prefetching

### Phase 3: Complete Redesign (1-2 months)

1. **Columnar Storage Format**
   - Design and implement columnar storage
   - Convert existing index to new format

2. **Thread-Local Index Copies**
   - Create thread-local copies of index data
   - Eliminate all synchronization in the critical path

3. **Lock-Free Parallel Processing**
   - Implement work stealing scheduler
   - Replace locks with atomic operations for result accumulation

4. **SIMD Acceleration Throughout**
   - Apply SIMD to all compute-intensive operations
   - Optimize for the available instruction sets

### Phase 4: Validation and Tuning (2-4 weeks)

1. **Performance Testing**
   - Benchmark against original implementation
   - Identify remaining bottlenecks

2. **Parameter Tuning**
   - Optimize batch sizes, thread counts, etc.
   - Fine-tune SIMD implementations

3. **Documentation and Integration**
   - Update documentation
   - Ensure backward compatibility

## Expected Results

With the proposed optimizations, we expect the following improvements:

1. **Search Time**: Reduction from 30+ seconds to 1-3 seconds (90-95% improvement)
2. **Ranking Phase**: Reduction from 28 seconds to under 1 second (95%+ improvement)
3. **Memory Usage**: Increased to 50-100GB (from current ~10GB), but well within the 256GB capacity
4. **Scalability**: Near-linear scaling with additional CPU cores

These improvements will dramatically enhance the user experience while maintaining the same search accuracy and functionality.
