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

#ifndef PASTEC_ORBINDEX_H
#define PASTEC_ORBINDEX_H

#include <fstream>
#include <string>
#include <memory>
#include <atomic>
#include <shared_mutex> // C++17 shared_mutex for reader/writer lock
#include <system_error>

#include <sys/types.h>

#include <map>
#include <vector>
#include <list>
#include <unordered_map>

#include <hit.h>
#include <backwardindexreaderaccess.h>
#include <index.h>

// Use of namespace std is kept for backward compatibility
using namespace std;

constexpr size_t NB_VISUAL_WORDS = 1000000;
constexpr size_t BACKWARD_INDEX_ENTRY_SIZE = 10;

class ORBIndex : public Index
{
public:
    ORBIndex(const std::string& indexPath, bool buildForwardIndex);
    virtual ~ORBIndex();
    
    // Updated method signatures to match the base class
    void getImagesWithVisualWords(const std::unordered_map<u_int32_t, std::vector<Hit>>& imagesReqHits,
                                  std::unordered_map<u_int32_t, std::vector<Hit>>& indexHitsForReq);
    
    unsigned getWordNbOccurences(unsigned i_wordId) const;
    unsigned countTotalNbWord(unsigned i_imageId) const;
    unsigned getTotalNbIndexedImages() const;
    
    u_int32_t addImage(unsigned i_imageId, const std::vector<HitForward>& hitList);
    u_int32_t addTag(const unsigned i_imageId, const string tag) override;
    u_int32_t removeImage(const unsigned i_imageId) override;
    u_int32_t getImageWords(unsigned i_imageId, std::unordered_map<u_int32_t, std::vector<Hit>>& hitList);
    u_int32_t removeTag(const unsigned i_imageId) override;
    u_int32_t getTag(unsigned i_imageId, string& tag) override;
    
    u_int32_t write(string backwardIndexPath) override;
    u_int32_t clear() override;
    u_int32_t load(string backwardIndexPath) override;
    u_int32_t getImageIds(vector<u_int32_t>& imageIds) override;

    u_int32_t loadTags(string indexTagsPath) override;
    u_int32_t writeTags(string indexTagsPath) override;

    // Thread-safe locking using C++17 shared_mutex
    void readLock();
    void unlock();

private:
    // Use a sparse representation for occurrence counts to save memory
    std::unordered_map<u_int32_t, u_int64_t> nbOccurences;
    std::atomic<u_int64_t> totalNbRecords;
    bool buildForwardIndex;

    // Improved data structures with consistent types
    std::unordered_map<u_int64_t, unsigned> nbWords;
    std::unordered_map<u_int64_t, std::vector<unsigned>> forwardIndex;
    std::unordered_map<u_int32_t, std::string> tags;
    
    // More memory-efficient sparse representation of visual words
    std::unordered_map<u_int32_t, std::vector<Hit>> indexHits;
    
    // Modern C++ threading primitives
    std::shared_mutex rwMutex;
    
    // Compatibility layer for older code
    pthread_rwlock_t rwLock;
};

#endif // PASTEC_ORBINDEX_H
