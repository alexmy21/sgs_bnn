include("sgs_entity.jl")

module Store
    using ..Entity
    using ..HllSets
    using ..Util

    using PyCall
    using CSV
    using DataFrames

    using JSON3
    using Base.Threads

    # Define the data store and version control structures
    mutable struct DataStore
        data::Dict{String, Any}
        commits::Vector{Dict{String, Any}}
        branches::Dict{String, Vector{Dict{String, Any}}}
        current_branch::String
    end

    # Initialize the data store
    function init_data_store()
        data = Dict{String, Any}()
        commits = Vector{Dict{String, Any}}()
        branches = Dict{String, Vector{Dict{String, Any}}}()
        branches["main"] = commits
        DataStore(data, commits, branches, "main")
    end

    # Commit the current state of the data with a commit message
    function commit(data_store::DataStore, message::String)
        commit_data = deepcopy(data_store.data)
        commit_entry = Dict("data" => commit_data, "message" => message, "timestamp" => now())
        push!(data_store.commits, commit_entry)
        println("Committed: $message")
    end

    # Checkout a previous commit by index
    function checkout(data_store::DataStore, commit_index::Int)
        if commit_index > 0 && commit_index <= length(data_store.commits)
            commit_entry = data_store.commits[commit_index]
            data_store.data = deepcopy(commit_entry["data"])
            println("Checked out commit: ", commit_entry["message"])
        else
            println("Invalid commit index")
        end
    end

    # Create a new branch
    function create_branch(data_store::DataStore, branch_name::String)
        data_store.branches[branch_name] = deepcopy(data_store.commits)
        println("Branch created: $branch_name")
    end

    # Merge a branch into the current branch
    function merge_branch(data_store::DataStore, branch_name::String)
        if haskey(data_store.branches, branch_name)
            branch_commits = data_store.branches[branch_name]
            data_store.commits = vcat(data_store.commits, branch_commits)
            println("Merged branch: $branch_name")
        else
            println("Branch not found: $branch_name")
        end
    end

    # CRUD functions
    # ====================================================================

    # Function to divide the array into N chunks
    function chunk_array(arr, N)
        chunks = []
        chunk_size = ceil(Int, length(arr) / N)
        for i in 1:chunk_size:length(arr)
            push!(chunks, arr[i:min(i + chunk_size - 1, length(arr))])
        end
        return chunks
    end

    # ingest DataFrame column by column
    #   - r:    A Ptthon object that represent Redis client
    #   - df:   The DataFrame containing the data to be ingested
    #   - cols: A vector of columns in df to be processed
    #   - p:    precision parameter for HllSet that defines the size of col_dataset

    function ingest_df(r::PyObject, tokenizer::PyObject, df::DataFrame, parent::String, cols::Vector; p::Int=10, chunk_size::Int=512000)
        
        for column in cols    
            col_values  = df[:, column]
            col_sha1    = Util.sha1_union([parent, string(column)])
            column_size = Base.summarysize(col_values)
            num_chunks  = ceil(Int, column_size / chunk_size)
            chunks      = chunk_array(col_values, num_chunks)

            println(col_sha1, "; num_chunks: ", num_chunks)
            dataset = ingest_df_column(r, tokenizer, chunks, col_sha1)
            println("Column dataset: ", dataset)
        end
    end

    function ingest_df_column(r::PyObject, tokenizer::PyObject, chunks, col_sha1::String; p::Int=10)
        # start = time()
        col_dataset = zeros(2^p)
        col_json    = JSON3.write(col_dataset)

        Threads.@sync begin 
            for chunk in chunks
                try
                    Threads.@spawn begin                    
                        # @info "$(chunk): Spawned $(time() - start)"
                        local_batch = Vector{String}()
                        for value in chunk
                            if value !== missing && value !== nothing
                                str_value   = string(value)
                                str_value   = String(str_value)                               
                                tokens      = tokenizer.tokenize(str_value)
                                append!(local_batch, tokens)
                            end
                        end
                        if !isempty(local_batch) 
                            unique_tokens   = Set(local_batch)
                            tokens          = JSON3.write(collect(unique_tokens))
                            # println("tokens: ", tokens)
                            result_json     = r.fcall("ingest_01", 2, "", col_sha1, p, tokens)
                            # println("result_json: ", typeof(result_json))
                            col_json        = r.fcall("bit_ops", 0, col_json, result_json, "OR")  
                            # println("col_json: ", (col_json))                      
                        end
                        # @info "$(chunk): Finished $(time() - start)"
                    end
                catch e 
                    println("chunk: ", chunk)
                end
            end
        end
        # println(col_json)
        return JSON3.read(col_json)
    end

    function ingest_df_rows(r::PyObject, tokenizer::PyObject, chunk, row_sha1::String; p::Int=10)
        println("processing row . . .")
    end
end