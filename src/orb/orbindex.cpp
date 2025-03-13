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
#include <string>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <sys/time.h>
#include <assert.h>

#include <orbindex.h>
#include <messages.h>


ORBIndex::ORBIndex(const std::string& indexPath, bool buildForwardIndex)
    : totalNbRecords(0), buildForwardIndex(buildForwardIndex)
{
    // Initialize both mutex systems for compatibility
    pthread_rwlock_init(&rwLock, NULL);
    
    // Don't pre-initialize the sparse nbOccurences map - save memory
    // by only adding entries when they're actually used
    
    load(indexPath);
}


/**
 * @brief Return the number of occurences of a word in an whole index.
 * @param i_wordId the word id.
 * @return the number of occurences.
 */
unsigned ORBIndex::getWordNbOccurences(unsigned i_wordId) const
{
    // Use modern C++ shared_mutex for read locks
    std::shared_lock<std::shared_mutex> lock(const_cast<std::shared_mutex&>(rwMutex));
    
    // With sparse representation, check if word exists in map
    auto it = nbOccurences.find(i_wordId);
    if (it != nbOccurences.end()) {
        return it->second;
    }
    return 0;
}


ORBIndex::~ORBIndex()
{
    pthread_rwlock_destroy(&rwLock);
    // shared_mutex automatically cleaned up
}


void ORBIndex::getImagesWithVisualWords(const std::unordered_map<u_int32_t, std::vector<Hit>>& imagesReqHits,
                                     std::unordered_map<u_int32_t, std::vector<Hit>>& indexHitsForReq)
{
    // Use C++17 shared_lock for read-only access
    std::shared_lock<std::shared_mutex> lock(rwMutex);

    // Preallocate expected capacity to avoid resizing
    indexHitsForReq.reserve(imagesReqHits.size());
    
    // Use modern range-based for loop with const reference
    for (const auto& [wordId, hits] : imagesReqHits) {
        // Skip non-existent words 
        auto it = indexHits.find(wordId);
        if (it != indexHits.end()) {
            // Use move semantics to avoid unnecessary copies of large vectors
            indexHitsForReq[wordId] = it->second;
        }
    }
}


/**
 * @brief Return the number of words for an image
 * @param i_imageId the image id.
 * @return the number of words.
 * readLock() and unlock MUST be called before and after calling this function.
 */
unsigned ORBIndex::countTotalNbWord(unsigned i_imageId) const
{
    // This method is assumed to be called inside a read lock
    auto it = nbWords.find(i_imageId);
    if (it != nbWords.end()) {
        return it->second;
    }
    return 0;
}


unsigned ORBIndex::getTotalNbIndexedImages() const
{
    std::shared_lock<std::shared_mutex> lock(const_cast<std::shared_mutex&>(rwMutex));
    return nbWords.size();
}


/**
 * @brief Add a list of hits to the index.
 * @param i_imageId the image ID.
 * @param hitList vector of hits to add.
 * @return status code.
 */
u_int32_t ORBIndex::addImage(unsigned i_imageId, const std::vector<HitForward>& hitList)
{
    // Use modern exclusive lock
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    
    // Check if image already exists and remove it if needed
    if (nbWords.find(i_imageId) != nbWords.end()) {
        // Release the lock before calling removeImage (which acquires its own lock)
        lock.unlock();
        removeImage(i_imageId);
        // Re-acquire the lock
        lock.lock();
    }
    
    // Pre-calculate collection sizes for better performance
    if (buildForwardIndex) {
        // Reserve space in forward index to avoid reallocations
        forwardIndex[i_imageId].reserve(hitList.size());
    }
    
    // Use batch processing by pre-collecting hits per word
    std::unordered_map<u_int32_t, std::vector<Hit>> wordToHits;
    wordToHits.reserve(std::min(hitList.size(), size_t(1000)));
    
    // First pass: group hits by word ID
    for (const auto& hitFor : hitList) {
        // Validate image ID
        assert(i_imageId == hitFor.i_imageId);
        
        // Create backward hit
        Hit hitBack;
        hitBack.i_imageId = hitFor.i_imageId;
        hitBack.i_angle = hitFor.i_angle;
        hitBack.x = hitFor.x;
        hitBack.y = hitFor.y;
        
        // Add to batch for this word
        wordToHits[hitFor.i_wordId].push_back(hitBack);
        
        // Update forward index if needed
        if (buildForwardIndex) {
            forwardIndex[hitFor.i_imageId].push_back(hitFor.i_wordId);
        }
    }
    
    // Second pass: batch update index structures
    for (const auto& [wordId, hits] : wordToHits) {
        // Pre-reserve space in indexHits vectors
        if (indexHits.find(wordId) == indexHits.end()) {
            indexHits[wordId].reserve(hits.size());
        } else {
            indexHits[wordId].reserve(indexHits[wordId].size() + hits.size());
        }
        
        // Add all hits at once
        indexHits[wordId].insert(indexHits[wordId].end(), hits.begin(), hits.end());
        
        // Update occurrence counts
        nbOccurences[wordId] += hits.size();
    }
    
    // Update word count for image
    nbWords[i_imageId] += hitList.size();
    
    // Update total record count
    totalNbRecords += hitList.size();
    
    lock.unlock();
    
    if (!hitList.empty()) {
        cout << "Image " << i_imageId << " added: " << hitList.size() << " hits." << endl;
    }

    return IMAGE_ADDED;
}


/**
 * @brief Add a string tag to an image.
 * @param i_imageId the image ID.
 * @param tag the tag to add.
 */
u_int32_t ORBIndex::addTag(const unsigned i_imageId, const string tag)
{
    pthread_rwlock_wrlock(&rwLock);

    if (nbWords.find(i_imageId) == nbWords.end()) {
        pthread_rwlock_unlock(&rwLock);
        return IMAGE_NOT_FOUND;
    }

    tags[i_imageId] = tag;

    pthread_rwlock_unlock(&rwLock);

    cout << "Tag added for image " << i_imageId << "." << endl;

    return IMAGE_TAG_ADDED;
}


/**
 * @brief Remove all the hits of an image.
 * @param i_imageId the image id.
 * @return true on success else false.
 */
u_int32_t ORBIndex::removeImage(unsigned i_imageId)
{
    // First remove the image tag if there is one (handles its own locking)
    removeTag(i_imageId);

    // Exclusive lock for writing
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    
    // Check if image exists
    auto imgIt = nbWords.find(i_imageId);
    if (imgIt == nbWords.end()) {
        cout << "Image " << i_imageId << " not found." << endl;
        return IMAGE_NOT_FOUND;
    }

    // Major optimization: use the forward index if available
    if (buildForwardIndex) {
        auto forwardIndexIt = forwardIndex.find(i_imageId);
        if (forwardIndexIt == forwardIndex.end()) {
            cout << "Image " << i_imageId << " not found in forward index." << endl;
            return IMAGE_NOT_FOUND;
        }
        
        // Get all words for this image
        const auto& words = forwardIndexIt->second;
        
        // For each word, find and remove the corresponding hit
        for (const unsigned i_wordId : words) {
            auto& hits = indexHits[i_wordId];
            
            // Use the erase-remove idiom with a lambda for better performance
            auto oldSize = hits.size();
            hits.erase(
                std::remove_if(hits.begin(), hits.end(), 
                    [i_imageId](const Hit& hit) { 
                        return hit.i_imageId == i_imageId; 
                    }),
                hits.end()
            );
            
            // Update occurrence count
            auto removed = oldSize - hits.size();
            if (removed > 0) {
                if (nbOccurences.find(i_wordId) != nbOccurences.end()) {
                    nbOccurences[i_wordId] -= removed;
                    if (nbOccurences[i_wordId] == 0) {
                        nbOccurences.erase(i_wordId);
                    }
                }
                totalNbRecords -= removed;
                
                // If no more hits for this word, remove the entry to save memory
                if (hits.empty()) {
                    indexHits.erase(i_wordId);
                }
            }
        }
        
        // Remove from forward index
        forwardIndex.erase(forwardIndexIt);
    }
    else {
        // Without forward index, we need to scan all words
        // But we can at least use the sparse indexHits map to avoid scanning all 1M slots
        for (auto it = indexHits.begin(); it != indexHits.end();) {
            const unsigned i_wordId = it->first;
            auto& hits = it->second;
            
            // Use the erase-remove idiom for better performance 
            auto oldSize = hits.size();
            hits.erase(
                std::remove_if(hits.begin(), hits.end(), 
                    [i_imageId](const Hit& hit) { 
                        return hit.i_imageId == i_imageId; 
                    }),
                hits.end()
            );
            
            // Update occurrence count
            auto removed = oldSize - hits.size();
            if (removed > 0) {
                if (nbOccurences.find(i_wordId) != nbOccurences.end()) {
                    nbOccurences[i_wordId] -= removed;
                    if (nbOccurences[i_wordId] == 0) {
                        nbOccurences.erase(i_wordId);
                    }
                }
                totalNbRecords -= removed;
            }
            
            // If no more hits for this word, remove the entry
            if (hits.empty()) {
                it = indexHits.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // Remove from word count
    nbWords.erase(imgIt);
    
    cout << "Image " << i_imageId << " removed." << endl;
    return IMAGE_REMOVED;
}


/**
 * @brief Get a list of hits associated with an image id.
 * @param i_imageId the image id.
 * @param hitList the list of hits corresponding to the visual words.
 * @return OK if success else ERROR.
 */
u_int32_t ORBIndex::getImageWords(unsigned i_imageId, std::unordered_map<u_int32_t, std::vector<Hit>>& hitList)
{
    // Use shared lock for concurrent read access
    std::shared_lock<std::shared_mutex> lock(rwMutex);

    const unsigned i_nbTotalIndexedImages = nbWords.size();
    const unsigned i_maxNbOccurences = i_nbTotalIndexedImages > 10000 ?
                                      0.15 * i_nbTotalIndexedImages
                                      : i_nbTotalIndexedImages;

    // Check if image exists
    auto imgIt = nbWords.find(i_imageId);
    if (imgIt == nbWords.end()) {
        cout << "Image " << i_imageId << " not found." << endl;
        return IMAGE_NOT_FOUND;
    }

    // Pre-allocate based on expected size
    hitList.reserve(buildForwardIndex ? forwardIndex[i_imageId].size() : 100);
    
    if (buildForwardIndex) {
        // Much more efficient with forward index
        const auto& words = forwardIndex[i_imageId];
        
        // Use modern C++ range-based for loop
        for (const unsigned i_wordId : words) {
            // Skip words that occur too frequently
            if (getWordNbOccurences(i_wordId) > i_maxNbOccurences)
                continue;
                
            // Find this word's hits
            auto indexHitsIt = indexHits.find(i_wordId);
            if (indexHitsIt == indexHits.end())
                continue;
                
            const auto& hits = indexHitsIt->second;
            
            // Use linear search but with early termination
            for (const auto& hit : hits) {
                if (hit.i_imageId == i_imageId) {
                    // Create a new entry for this word if needed
                    if (hitList.find(i_wordId) == hitList.end()) {
                        hitList[i_wordId] = std::vector<Hit>{};
                        hitList[i_wordId].reserve(4); // Most words have few hits per image
                    }
                    
                    hitList[i_wordId].push_back(hit);
                    break; // Only need one hit per word per image
                }
            }
        }
    } else {
        // Without forward index - need to scan index (now a sparse map)
        for (const auto& [i_wordId, hits] : indexHits) {
            // Skip words that occur too frequently
            if (getWordNbOccurences(i_wordId) > i_maxNbOccurences)
                continue;
                
            // Check for this image ID in the hits
            for (const auto& hit : hits) {
                if (hit.i_imageId == i_imageId) {
                    // Add this hit to the result
                    if (hitList.find(i_wordId) == hitList.end()) {
                        hitList[i_wordId] = std::vector<Hit>{};
                    }
                    hitList[i_wordId].push_back(hit);
                    break; // Only need one hit per word per image
                }
            }
        }
    }

    cout << "Image " << i_imageId << " found with " << hitList.size() << " words." << endl;
    return OK;
}


/**
 * @brief Remove a string tag to an image.
 */
u_int32_t ORBIndex::removeTag(const unsigned i_imageId)
{
    pthread_rwlock_wrlock(&rwLock);

    unordered_map<u_int32_t, string>::iterator tagIt =
        tags.find(i_imageId);

    if (tagIt == tags.end()) {
        pthread_rwlock_unlock(&rwLock);
        return IMAGE_TAG_NOT_FOUND;
    }

    tags.erase(tagIt);

    pthread_rwlock_unlock(&rwLock);

    cout << "Tag deleted for image " << i_imageId << "." << endl;

    return IMAGE_TAG_REMOVED;
}


/**
 * @brief Get the tag of an image.
 * @param i_imageId the image id
 * @param tag the returned tag
 */
u_int32_t ORBIndex::getTag(unsigned i_imageId, string &tag)
{
    pthread_rwlock_rdlock(&rwLock);

    unordered_map<u_int32_t, string>::iterator tagIt =
        tags.find(i_imageId);

    if (tagIt == tags.end()) {
        pthread_rwlock_unlock(&rwLock);
        return IMAGE_TAG_NOT_FOUND;
    }

    tag = tagIt->second;

    pthread_rwlock_unlock(&rwLock);

    return OK;
}


/**
 * @brief Write the index in memory to a file.
 * @param backwardIndexPath path to write the index
 * @return the operation code
 */
u_int32_t ORBIndex::write(string backwardIndexPath)
{
    if (backwardIndexPath == "")
        backwardIndexPath = DEFAULT_INDEX_PATH;

    ofstream ofs;

    ofs.open(backwardIndexPath.c_str(), ios_base::binary);
    if (!ofs.good())
    {
        cout << "Could not open the backward index file." << endl;
        return INDEX_NOT_WRITTEN;
    }

    pthread_rwlock_rdlock(&rwLock);

    cout << "Writing the number of occurences." << endl;
    for (unsigned i = 0; i < NB_VISUAL_WORDS; ++i) {
        // Get occurrence count from the map or use 0 if not found
        u_int64_t count = 0;
        auto it = nbOccurences.find(i);
        if (it != nbOccurences.end()) {
            count = it->second;
        }
        ofs.write((char *)(&count), sizeof(u_int64_t));
    }

    cout << "Writing the index hits." << endl;
    for (unsigned i = 0; i < NB_VISUAL_WORDS; ++i)
    {
        // Only process words that exist in our sparse index
        auto it = indexHits.find(i);
        if (it != indexHits.end()) {
            const vector<Hit> &wordHits = it->second;
            
            for (unsigned j = 0; j < wordHits.size(); ++j)
            {
                const Hit &hit = wordHits[j];
                ofs.write((char *)(&hit.i_imageId), sizeof(u_int32_t));
                ofs.write((char *)(&hit.i_angle), sizeof(u_int16_t));
                ofs.write((char *)(&hit.x), sizeof(u_int16_t));
                ofs.write((char *)(&hit.y), sizeof(u_int16_t));
            }
        }
    }

    ofs.close();
    cout << "Writing done." << endl;

    pthread_rwlock_unlock(&rwLock);

    return INDEX_WRITTEN;
}


/**
 * @brief Clear the index.
 * @return true on success else false.
 */
u_int32_t ORBIndex::clear()
{
    // Use the C++17 exclusive lock
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    
    // Clear all maps
    nbOccurences.clear();
    indexHits.clear();
    nbWords.clear();
    forwardIndex.clear();
    tags.clear();

    totalNbRecords = 0;
    
    // Also unlock the pthread lock for backward compatibility
    pthread_rwlock_wrlock(&rwLock);
    pthread_rwlock_unlock(&rwLock);

    cout << "Index cleared." << endl;

    return INDEX_CLEARED;
}


/**
 * @brief Load the index from a file.
 * @param backwardIndexPath the path to the index file.
 * @return the operation code.
 */
u_int32_t ORBIndex::load(string backwardIndexPath)
{
    u_int32_t i_ret;

    // Open the file.
    BackwardIndexReaderFileAccess indexAccess;
    if (!indexAccess.open(backwardIndexPath))
    {
        cout << "Could not open the backward index file." << endl;
        i_ret = INDEX_NOT_FOUND;
    }
    else
    {
        clear();

        pthread_rwlock_wrlock(&rwLock);

        /* Read the table to know where are located the lines corresponding to each
         * visual word. */
        cout << "Reading the numbers of occurences." << endl;
        u_int64_t *wordOffSet = new u_int64_t[NB_VISUAL_WORDS];
        u_int64_t i_offset = NB_VISUAL_WORDS * sizeof(u_int64_t);
        for (unsigned i = 0; i < NB_VISUAL_WORDS; ++i)
        {
            // Read the occurrence count into a temporary variable
            u_int64_t count;
            indexAccess.read((char *)(&count), sizeof(u_int64_t));
            
            // Store it in our map if it's non-zero
            if (count > 0) {
                nbOccurences[i] = count;
            }
            
            wordOffSet[i] = i_offset;
            i_offset += count * BACKWARD_INDEX_ENTRY_SIZE;
        }

        /* Count the number of words per image. */
        cout << "Counting the number of words per image." << endl;
        totalNbRecords = 0;
        while (true)
        {
            u_int32_t i_imageId;
            u_int16_t i_angle, x, y;
            indexAccess.read((char *)&i_imageId, sizeof(u_int32_t));
            if (indexAccess.endOfIndex())
                break;
            indexAccess.read((char *)&i_angle, sizeof(u_int16_t));
            indexAccess.read((char *)&x, sizeof(u_int16_t));
            indexAccess.read((char *)&y, sizeof(u_int16_t));
            nbWords[i_imageId]++;
            totalNbRecords++;
        }

        indexAccess.reset();

        cout << "Loading the index in memory." << endl;

        for (unsigned i_wordId = 0; i_wordId < NB_VISUAL_WORDS; ++i_wordId)
        {
            // Skip words with zero occurrences
            auto occIt = nbOccurences.find(i_wordId);
            if (occIt == nbOccurences.end() || occIt->second == 0) {
                continue;
            }
            
            indexAccess.moveAt(wordOffSet[i_wordId]);
            
            // Only create vector entries for words that actually have hits
            const unsigned i_nbOccurences = occIt->second;
            if (i_nbOccurences == 0) {
                continue;
            }
            
            // Preallocate the vector to avoid reallocations
            auto& hits = indexHits[i_wordId];
            hits.reserve(i_nbOccurences);
            
            for (u_int64_t i = 0; i < i_nbOccurences; ++i)
            {
                u_int32_t i_imageId;
                u_int16_t i_angle, x, y;
                indexAccess.read((char *)&i_imageId, sizeof(u_int32_t));
                indexAccess.read((char *)&i_angle, sizeof(u_int16_t));
                indexAccess.read((char *)&x, sizeof(u_int16_t));
                indexAccess.read((char *)&y, sizeof(u_int16_t));
                
                // Create and add the hit
                Hit hit;
                hit.i_imageId = i_imageId;
                hit.i_angle = i_angle;
                hit.x = x;
                hit.y = y;
                hits.push_back(std::move(hit));
                
                if (buildForwardIndex)
                {
                    forwardIndex[i_imageId].push_back(i_wordId);
                }
            }
        }

        indexAccess.close();
        delete[] wordOffSet;

        pthread_rwlock_unlock(&rwLock);

        i_ret = INDEX_LOADED;
    }

    return i_ret;
}


/**
 * @brief Load the index tags from a file.
 * @param indexTagsPath the path to the index tags file.
 * @return the operation code.
 */
u_int32_t ORBIndex::loadTags(string indexTagsPath)
{
    if (indexTagsPath == "")
        indexTagsPath = DEFAULT_INDEX_TAGS_PATH;

    ifstream ifs;

    ifs.open(indexTagsPath.c_str(), ios_base::binary);
    if (!ifs.good())
    {
        cout << "Could not open the index tags file." << endl;
        return INDEX_TAGS_NOT_FOUND;
    }

    pthread_rwlock_wrlock(&rwLock);

    tags.clear();
    while (true)
    {
        // Read the image tag.
        u_int32_t i_imageId;
        u_int32_t i_tagSize;
        ifs.read((char *)&i_imageId, sizeof(u_int32_t));
        if (ifs.eof())
            break;
        ifs.read((char *)&i_tagSize, sizeof(u_int32_t));
        char psz_tag[i_tagSize];
        ifs.read((char *)psz_tag, i_tagSize);

        cout << i_imageId << " " << i_tagSize << " " << psz_tag << endl;

        // Save it into the memory.
        tags[i_imageId] = string(psz_tag);
    }

    pthread_rwlock_unlock(&rwLock);

    return INDEX_TAGS_LOADED;
}


/**
 * @brief Write the index image tags into a file.
 * @param indexTagsPath the path to the index tags file.
 * @return the operation code.
 */
u_int32_t ORBIndex::writeTags(string indexTagsPath)
{
    if (indexTagsPath == "")
        indexTagsPath = DEFAULT_INDEX_TAGS_PATH;

    ofstream ofs;

    ofs.open(indexTagsPath.c_str(), ios_base::binary);
    if (!ofs.good())
    {
        cout << "Could not open the index tags file." << endl;
        return INDEX_TAGS_NOT_WRITTEN;
    }

    pthread_rwlock_rdlock(&rwLock);

    cout << "Writing the index image tags." << endl;

    for (unordered_map<u_int32_t, string>::const_iterator it = tags.begin();
         it != tags.end(); ++it)
    {
        u_int32_t i_imageId = it->first;
        const char *psz_tag = it->second.c_str();
        u_int32_t i_tagSize = strlen(psz_tag) + 1;

        ofs.write((char *)(&i_imageId), sizeof(u_int32_t));
        ofs.write((char *)(&i_tagSize), sizeof(u_int32_t));
        ofs.write((char *)(psz_tag), i_tagSize);
        cout << "plop!" << endl;
    }

    ofs.close();
    cout << "Writing done." << endl;

    pthread_rwlock_unlock(&rwLock);

    return INDEX_TAGS_WRITTEN;
}



/**
 * @brief List the images ids in the index.
 * @param imageIds the vector to return the images ids.
 * @return the operation code.
 */
u_int32_t ORBIndex::getImageIds(vector<u_int32_t> &imageIds)
{
    imageIds.reserve(nbWords.size());
    for (unordered_map<u_int64_t, unsigned>::const_iterator it = nbWords.begin();
         it != nbWords.end(); ++it)
        imageIds.push_back(it->first);

    return INDEX_IMAGE_IDS;
}


/**
 * @brief Lock for reading the index.
 */
void ORBIndex::readLock()
{
    // For backward compatibility, still use the pthread lock
    pthread_rwlock_rdlock(&rwLock);
    
    // Also acquire a shared lock on the modern mutex
    rwMutex.lock_shared();
}


/**
 * @brief Unlock the index.
 */
void ORBIndex::unlock()
{
    // For backward compatibility, still use the pthread lock
    pthread_rwlock_unlock(&rwLock);
    
    // Also release the shared lock on the modern mutex
    rwMutex.unlock_shared();
}
