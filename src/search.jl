include("sets32.jl")
include("utils.jl")

module Search
    using ..HllSets
    using ..Util

    # using TidierDB
    using Redis
    using JSON3: JSON3
    using EasyConfig
    using DataFrames: DataFrame
    using DataFrames: DataFrameRow

    export vec_query_tokens, vec_query_tokens_id, vec_query_nodes, 
            token_id_from_text, token_id_from_node_ids,             
            node_refs_from_text, node_refs_from_tokens, node_refs_from_tokens_ids

    function vec_query_tokens(tokens::Set{String})
        hll = HllSets.HllSet{8}()
        for token in tokens
            HllSets.add!(hll, HllSets.u_hash(token))
        end        
        return HllSets.dump(hll)
    end

    function vec_query_tokens_id(tokens::Set{UInt64})
        hll = HllSets.HllSet{8}()
        for token in tokens
            HllSets.add!(hll, token)
        end        
        return Util.to_blob(HllSets.dump(hll))
    end

    function vec_query_tokens(tokens::String...)
        hll = HllSets.HllSet{8}()
        for token in tokens
            # Assuming token is the ID needed to fetch from Redis
            HllSets.add!(hll, token)
        end
        return Util.to_blob(HllSets.dump(hll))
    end

    # Search functions
    #-------------------------------------------------
    function token_id_from_text(conn::RedisConnection, idx_name::String, text::String)
        text_tokens = split(text)
        vec_query = vec_query_tokens(Set(text_tokens))

        return Redis.execute_command(conn, ["FT.SEARCH", "$idx_name", "*"])
    end
end