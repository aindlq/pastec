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


ORBIndex::ORBIndex(string indexPath, string tagsPath, bool buildForwardIndex)
    : buildForwardIndex(buildForwardIndex), totalNbRecords(0),
      storedIndexPath(indexPath), storedTagsPath(tagsPath)
{
    // Init the mutex.
    pthread_rwlock_init(&rwLock, NULL);

    // Initialize the nbOccurences table.
    for (unsigned i = 0; i < NB_VISUAL_WORDS; ++i)
        nbOccurences[i] = 0;

    load(indexPath);
    loadTags(tagsPath);
    cout << "DEBUG: Index initialized with " << nbWords.size() << endl;
}


/**
 * @brief Return the number of occurences of a word in an whole index.
 * @param i_wordId the word id.
 * @return the number of occurences.
 */
unsigned ORBIndex::getWordNbOccurences(unsigned i_wordId)
{
    // No locks needed since the index is read-only during queries
    assert(i_wordId < NB_VISUAL_WORDS);
    unsigned i_ret = nbOccurences[i_wordId];
    return i_ret;
}


ORBIndex::~ORBIndex()
{
    pthread_rwlock_destroy(&rwLock);
}


void ORBIndex::getImagesWithVisualWords(unordered_map<u_int32_t, list<Hit> > &imagesReqHits,
                                     unordered_map<u_int32_t, const vector<Hit>* > &indexHitsForReq)
{
    // Pre-allocate memory for the result to avoid reallocations
    indexHitsForReq.reserve(imagesReqHits.size());
    
    // No locks needed since the index is read-only during queries
    for (unordered_map<u_int32_t, list<Hit> >::const_iterator it = imagesReqHits.begin();
         it != imagesReqHits.end(); ++it)
    {
        const unsigned i_wordId = it->first;
        
        // Direct access without locks
        const vector<Hit>& hits = indexHits[i_wordId];
        
        // Use references instead of deep copies
        indexHitsForReq[i_wordId] = &hits;
    }
}


/**
 * @brief Return the number of words for an image
 * @param i_imageId the image id.
 * @return the number of words.
 * No locks needed since the index is read-only during queries.
 */
unsigned ORBIndex::countTotalNbWord(unsigned i_imageId)
{
    // Make sure the image ID is within bounds
    if (i_imageId >= nbWords.size())
        return 0;
    
    unsigned i_ret = nbWords[i_imageId];
    return i_ret;
}


unsigned ORBIndex::getTotalNbIndexedImages()
{
    // No locks needed since the index is read-only during queries
    // Count non-zero entries in the nbWords vector
    unsigned count = 0;
    for (const auto& wordCount : nbWords) {
        if (wordCount > 0) {
            count++;
        }
    }
    return count;
}


/**
 * @brief Add a list of hits to the index.
 * @param  the list of hits.
 */
u_int32_t ORBIndex::addImage(unsigned i_imageId, list<HitForward> hitList)
{
    pthread_rwlock_wrlock(&rwLock);
    
    // Check if image already exists
    if (i_imageId < nbWords.size() && nbWords[i_imageId] > 0)
    {
        pthread_rwlock_unlock(&rwLock);
        removeImage(i_imageId);
        pthread_rwlock_wrlock(&rwLock);
    }
    
    // Ensure vectors have sufficient capacity
    if (i_imageId >= nbWords.size())
    {
        nbWords.resize(i_imageId + 1, 0);
        if (buildForwardIndex)
        {
            forwardIndex.resize(i_imageId + 1);
        }
        tags.resize(i_imageId + 1);
    }

    for (list<HitForward>::iterator it = hitList.begin(); it != hitList.end(); ++it)
    {
        HitForward hitFor = *it;
        assert(i_imageId == hitFor.i_imageId);
        Hit hitBack;
        hitBack.i_imageId = hitFor.i_imageId;
        hitBack.i_angle = hitFor.i_angle;
        hitBack.x = hitFor.x;
        hitBack.y = hitFor.y;

        if (buildForwardIndex)
        {
            forwardIndex[hitFor.i_imageId].push_back(hitFor.i_wordId);
        }
        indexHits[hitFor.i_wordId].push_back(hitBack);
        nbWords[hitFor.i_imageId]++;
        nbOccurences[hitFor.i_wordId]++;
        totalNbRecords++;
    }
    pthread_rwlock_unlock(&rwLock);

    if (!hitList.empty())
        cout << "Image " << hitList.begin()->i_imageId << " added: "
             << hitList.size() << " hits." << endl;

    return IMAGE_ADDED;
}

/**
 * @brief Add multiple images to the index in a single transaction.
 * @param batchHits map of image IDs to their respective hit lists.
 * @return IMAGE_ADDED on success.
 */
u_int32_t ORBIndex::addBatchImages(const unordered_map<u_int32_t, list<HitForward>>& batchHits)
{
    pthread_rwlock_wrlock(&rwLock);
    
    // Find maximum image ID to ensure array capacity
    u_int32_t maxImageId = 0;
    for (const auto& pair : batchHits) {
        maxImageId = std::max(maxImageId, pair.first);
    }
    
    // Ensure vectors have sufficient capacity
    if (maxImageId >= nbWords.size()) {
        nbWords.resize(maxImageId + 1, 0);
        if (buildForwardIndex) {
            forwardIndex.resize(maxImageId + 1);
        }
        tags.resize(maxImageId + 1);
    }
    
    // Add all the new hits
    for (const auto& pair : batchHits) {
        u_int32_t imageId = pair.first;
        const list<HitForward>& hitList = pair.second;
        
        for (const HitForward& hitFor : hitList) {
            assert(imageId == hitFor.i_imageId);
            Hit hitBack;
            hitBack.i_imageId = hitFor.i_imageId;
            hitBack.i_angle = hitFor.i_angle;
            hitBack.x = hitFor.x;
            hitBack.y = hitFor.y;
            
            if (buildForwardIndex) {
                forwardIndex[hitFor.i_imageId].push_back(hitFor.i_wordId);
            }
            indexHits[hitFor.i_wordId].push_back(hitBack);
            nbWords[hitFor.i_imageId]++;
            nbOccurences[hitFor.i_wordId]++;
            totalNbRecords++;
        }
    }
    
    pthread_rwlock_unlock(&rwLock);
    
    return IMAGE_ADDED;
}

/**
 * @brief Add multiple tags to images in a single transaction.
 * @param batchTags map of image IDs to their respective tags.
 * @return IMAGE_TAG_ADDED on success.
 */
u_int32_t ORBIndex::addBatchTags(const unordered_map<u_int32_t, string>& batchTags)
{
    if (batchTags.empty()) {
        cout << "DEBUG: No tags to add in batch" << endl;
        return OK;
    }
    
    pthread_rwlock_wrlock(&rwLock);
    
    // Find maximum image ID to ensure array capacity
    u_int32_t maxImageId = 0;
    for (const auto& pair : batchTags) {
        maxImageId = std::max(maxImageId, pair.first);
    }
    
    // Ensure tags vector has sufficient capacity
    if (maxImageId >= tags.size()) {
        tags.resize(maxImageId + 1);
    }
    
    // Add all tags
    for (const auto& pair : batchTags) {
        tags[pair.first] = pair.second;
    }
    
    pthread_rwlock_unlock(&rwLock);
        
    return IMAGE_TAG_ADDED;
}


/**
 * @brief Add a string tag to an image.
 * @param  the tag to add.
 */
u_int32_t ORBIndex::addTag(const unsigned i_imageId, const string tag)
{
    pthread_rwlock_wrlock(&rwLock);

    // Check if image exists
    if (i_imageId >= nbWords.size() || nbWords[i_imageId] == 0) {
        pthread_rwlock_unlock(&rwLock);
        return IMAGE_NOT_FOUND;
    }

    // Ensure tags vector has sufficient capacity
    if (i_imageId >= tags.size()) {
        tags.resize(i_imageId + 1);
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
u_int32_t ORBIndex::removeImage(const unsigned i_imageId)
{
    // First remove the image tag if there is one.
    removeTag((u_int64_t)i_imageId);

    pthread_rwlock_wrlock(&rwLock);
    
    // Check if image exists
    if (i_imageId >= nbWords.size() || nbWords[i_imageId] == 0)
    {
        cout << "Image " << i_imageId << " not found." << endl;
        pthread_rwlock_unlock(&rwLock);
        return IMAGE_NOT_FOUND;
    }

    // Set word count to 0 (effectively removing the image)
    nbWords[i_imageId] = 0;

    if (buildForwardIndex && i_imageId < forwardIndex.size())
    {
        // Clear the forward index for this image
        forwardIndex[i_imageId].clear();
    }

    // Remove hits from indexHits
    for (unsigned i_wordId = 0; i_wordId < NB_VISUAL_WORDS; ++i_wordId)
    {
        vector<Hit> &hits = indexHits[i_wordId];
        vector<Hit>::iterator it = hits.begin();

        while (it != hits.end())
        {
            if (it->i_imageId == i_imageId)
            {
                totalNbRecords--;
                nbOccurences[i_wordId]--;
                hits.erase(it);
                break;
            }
            ++it;
        }
    }
    pthread_rwlock_unlock(&rwLock);

    cout << "Image " << i_imageId << " removed." << endl;

    return IMAGE_REMOVED;
}


/**
 * @brief Get a list of hits associated with an image id.
 * @param  the list of hits.
 */
u_int32_t ORBIndex::getImageWords(unsigned i_imageId, unordered_map<u_int32_t, list<Hit> > &hitList)
{
    // No locks needed since the index is read-only during queries
    const unsigned i_nbTotalIndexedImages = getTotalNbIndexedImages();
    const unsigned i_maxNbOccurences = i_nbTotalIndexedImages > 10000 ?
                                       0.15 * i_nbTotalIndexedImages
                                       : i_nbTotalIndexedImages;

    // Check if image exists
    if (i_imageId >= nbWords.size() || nbWords[i_imageId] == 0)
    {
        cout << "Image " << i_imageId << " not found." << endl;
        return IMAGE_NOT_FOUND;
    }

    if (buildForwardIndex && i_imageId < forwardIndex.size())
    {
        const vector<unsigned> &words = forwardIndex[i_imageId];
        
        for (const unsigned i_wordId : words)
        {
            if (getWordNbOccurences(i_wordId) <= i_maxNbOccurences)
            {
                vector<Hit> &hits = indexHits[i_wordId];
                
                for (const Hit &hit : hits)
                {
                    if (hit.i_imageId == i_imageId)
                    {
                        hitList[i_wordId].push_back(hit);
                        break;
                    }
                }
            }
        }
    }
    else
    {
        for (unsigned i_wordId = 0; i_wordId < NB_VISUAL_WORDS; ++i_wordId)
        {
            vector<Hit> &hits = indexHits[i_wordId];
            
            for (const Hit &hit : hits)
            {
                if (hit.i_imageId == i_imageId)
                {
                    if (getWordNbOccurences(i_wordId) <= i_maxNbOccurences)
                    {
                        hitList[i_wordId].push_back(hit);
                    }
                    break;
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

    // Check if tag exists
    if (i_imageId >= tags.size() || tags[i_imageId].empty()) {
        pthread_rwlock_unlock(&rwLock);
        return IMAGE_TAG_NOT_FOUND;
    }

    // Clear the tag (set to empty string)
    tags[i_imageId] = "";

    pthread_rwlock_unlock(&rwLock);

    cout << "Tag deleted for image " << i_imageId << "." << endl;

    return IMAGE_TAG_REMOVED;
}


/**
 * @brief Get the tag of an image.
 * @param the image id,
 * @param the returned tag.
 */
u_int32_t ORBIndex::getTag(const unsigned i_imageId, string &tag)
{
    // No locks needed since the index is read-only during queries
    
    // Check if tag exists
    if (i_imageId >= tags.size() || tags[i_imageId].empty()) {
        return IMAGE_TAG_NOT_FOUND;
    }

    tag = tags[i_imageId];

    return OK;
}


/**
 * @brief Write the index in memory to a file.
 * @param backwardIndexPath
 * @return the operation code
 */
u_int32_t ORBIndex::write(string backwardIndexPath)
{
    if (backwardIndexPath == "")
    {
        // If no path is provided, use the stored path from constructor
        if (storedIndexPath != "")
            backwardIndexPath = storedIndexPath;
        else
            backwardIndexPath = DEFAULT_INDEX_PATH;
    }

    ofstream ofs;

    ofs.open(backwardIndexPath.c_str(), ios_base::binary);
    if (!ofs.good())
    {
        cout << "Could not open the backward index file." << endl;
        return INDEX_NOT_WRITTEN;
    }

    pthread_rwlock_rdlock(&rwLock);

    cout << "Writing the number of occurences." << endl;
    for (unsigned i = 0; i < NB_VISUAL_WORDS; ++i)
        ofs.write((char *)(nbOccurences + i), sizeof(u_int64_t));

    cout << "Writing the index hits." << endl;
    for (unsigned i = 0; i < NB_VISUAL_WORDS; ++i)
    {
        const vector<Hit> &wordHits = indexHits[i];

        for (unsigned j = 0; j < wordHits.size(); ++j)
        {
            const Hit &hit = wordHits[j];
            ofs.write((char *)(&hit.i_imageId), sizeof(u_int32_t));
            ofs.write((char *)(&hit.i_angle), sizeof(u_int16_t));
            ofs.write((char *)(&hit.x), sizeof(u_int16_t));
            ofs.write((char *)(&hit.y), sizeof(u_int16_t));
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
    pthread_rwlock_wrlock(&rwLock);
    
    // Reset the nbOccurences table.
    for (unsigned i = 0; i < NB_VISUAL_WORDS; ++i)
    {
        nbOccurences[i] = 0;
        indexHits[i].clear();
    }

    // Clear vectors (but keep their capacity)
    std::fill(nbWords.begin(), nbWords.end(), 0);
    for (auto& words : forwardIndex) {
        words.clear();
    }
    std::fill(tags.begin(), tags.end(), "");

    totalNbRecords = 0;
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
            indexAccess.read((char *)(nbOccurences + i), sizeof(u_int64_t));
            wordOffSet[i] = i_offset;
            i_offset += nbOccurences[i] * BACKWARD_INDEX_ENTRY_SIZE;
        }

        /* First pass: find maximum image ID. */
        cout << "Finding maximum image ID." << endl;
        u_int32_t maxImageId = 0;
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
            maxImageId = std::max(maxImageId, i_imageId);
        }
        
        // Ensure vectors have sufficient capacity
        nbWords.resize(maxImageId + 1, 0);
        if (buildForwardIndex)
        {
            forwardIndex.resize(maxImageId + 1);
        }
        
        /* Second pass: count the number of words per image. */
        cout << "Counting the number of words per image." << endl;
        indexAccess.reset();
        indexAccess.moveAt(NB_VISUAL_WORDS * sizeof(u_int64_t)); // Skip nbOccurences
        
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
            indexAccess.moveAt(wordOffSet[i_wordId]);
            vector<Hit> &hits = indexHits[i_wordId];

            const unsigned i_nbOccurences = nbOccurences[i_wordId];
            hits.resize(i_nbOccurences);

            for (u_int64_t i = 0; i < i_nbOccurences; ++i)
            {
                u_int32_t i_imageId;
                u_int16_t i_angle, x, y;
                indexAccess.read((char *)&i_imageId, sizeof(u_int32_t));
                indexAccess.read((char *)&i_angle, sizeof(u_int16_t));
                indexAccess.read((char *)&x, sizeof(u_int16_t));
                indexAccess.read((char *)&y, sizeof(u_int16_t));
                hits[i].i_imageId = i_imageId;
                hits[i].i_angle = i_angle;
                hits[i].x = x;
                hits[i].y = y;

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

    // First pass: find maximum image ID
    u_int32_t maxImageId = 0;
    ifstream firstPassIfs(indexTagsPath.c_str(), ios_base::binary);
    
    while (true)
    {
        u_int32_t i_imageId;
        firstPassIfs.read((char *)&i_imageId, sizeof(u_int32_t));
        if (firstPassIfs.eof())
            break;
            
        // Skip tag size and tag content
        u_int32_t i_tagSize;
        firstPassIfs.read((char *)&i_tagSize, sizeof(u_int32_t));
        firstPassIfs.seekg(i_tagSize, ios_base::cur);
        
        maxImageId = std::max(maxImageId, i_imageId);
    }
    
    firstPassIfs.close();
    
    // Ensure tags vector has sufficient capacity
    tags.resize(maxImageId + 1);
    
    // Second pass: load the actual tags
    ifs.clear();
    ifs.seekg(0, ios_base::beg);
    
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
    {
        // If no path is provided, use the stored path from constructor
        if (storedTagsPath != "")
            indexTagsPath = storedTagsPath;
        else
            indexTagsPath = DEFAULT_INDEX_TAGS_PATH;
    }

    ofstream ofs;

    ofs.open(indexTagsPath.c_str(), ios_base::binary);
    if (!ofs.good())
    {
        cout << "Could not open the index tags file." << endl;
        return INDEX_TAGS_NOT_WRITTEN;
    }

    pthread_rwlock_rdlock(&rwLock);

    cout << "Writing the index image tags." << endl;

    // Write only non-empty tags
    for (size_t i_imageId = 0; i_imageId < tags.size(); ++i_imageId)
    {
        if (!tags[i_imageId].empty())
        {
            const char *psz_tag = tags[i_imageId].c_str();
            u_int32_t i_tagSize = strlen(psz_tag) + 1;

            ofs.write((char *)(&i_imageId), sizeof(u_int32_t));
            ofs.write((char *)(&i_tagSize), sizeof(u_int32_t));
            ofs.write((char *)(psz_tag), i_tagSize);
        }
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
    // No locks needed since the index is read-only during queries
    
    // Count non-zero entries first to reserve the right amount of space
    unsigned count = 0;
    for (size_t i = 0; i < nbWords.size(); ++i) {
        if (nbWords[i] > 0) {
            count++;
        }
    }
    
    imageIds.reserve(count);
    
    // Add all image IDs with non-zero word counts
    for (size_t i = 0; i < nbWords.size(); ++i) {
        if (nbWords[i] > 0) {
            imageIds.push_back(i);
        }
    }

    return INDEX_IMAGE_IDS;
}


/**
 * @brief Lock for reading the index.
 */
void ORBIndex::readLock()
{
    pthread_rwlock_rdlock(&rwLock);
}


/**
 * @brief Unlock the index.
 */
void ORBIndex::unlock()
{
    pthread_rwlock_unlock(&rwLock);
}

/**
 * @brief Get direct access to the word count vector.
 * @return A const reference to the nbWords vector.
 */
const vector<unsigned>& ORBIndex::getWordCountVector() const
{
    return nbWords;
}
