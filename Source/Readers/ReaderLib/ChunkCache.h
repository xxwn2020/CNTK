//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <map>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ChunkCache : public IDataDeserializer
{
public:

    ChunkCache(IDataDeserializerPtr deserializer) : m_deserializer(deserializer) { }

    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_deserializer->GetStreamDescriptions();
    }

    virtual ChunkDescriptions GetChunkDescriptions() override
    {
        return m_deserializer->GetChunkDescriptions();
    }

    virtual void GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& descriptions) override
    {
        return m_deserializer->GetSequencesForChunk(chunkId, descriptions);
    }

    virtual void GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& description) override
    {
        return m_deserializer->GetSequenceDescriptionByKey(key, description);
    }

    // Gets chunk data given its id.
    virtual ChunkPtr GetChunk(size_t chunkId);

private:
    // A map of currently loaded chunks
    std::map<size_t, ChunkPtr> m_chunkMap;
    IDataDeserializerPtr m_deserializer;

    DISABLE_COPY_AND_MOVE(ChunkCache);
};

} } }