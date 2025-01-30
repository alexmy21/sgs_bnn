#!lua name=my_lib

-- MurmurHash3 implementation in Lua
local function murmurhash3_32(key, seed)
    local c1 = 0xcc9e2d51
    local c2 = 0x1b873593
    local r1 = 15
    local r2 = 13
    local m = 5
    local n = 0xe6546b64

    local hash = seed

    local len = #key
    local roundedEnd = bit.band(len, 0xfffffffc) -- round down to 4 byte block

    for i = 1, roundedEnd, 4 do
        local k = bit.bor(
            string.byte(key, i),
            bit.lshift(string.byte(key, i + 1), 8),
            bit.lshift(string.byte(key, i + 2), 16),
            bit.lshift(string.byte(key, i + 3), 24)
        )
        k = k * c1
        k = bit.bor(bit.lshift(k, r1), bit.rshift(k, 32 - r1))
        k = k * c2

        hash = bit.bxor(hash, k)
        hash = bit.bor(bit.lshift(hash, r2), bit.rshift(hash, 32 - r2))
        hash = hash * m + n
    end

    local k = 0
    if bit.band(len, 0x03) == 3 then
        k = bit.lshift(string.byte(key, roundedEnd + 2), 16)
    end
    if bit.band(len, 0x03) >= 2 then
        k = bit.bor(k, bit.lshift(string.byte(key, roundedEnd + 1), 8))
    end
    if bit.band(len, 0x03) >= 1 then
        k = bit.bor(k, string.byte(key, roundedEnd))
        k = k * c1
        k = bit.bor(bit.lshift(k, r1), bit.rshift(k, 32 - r1))
        k = k * c2
        hash = bit.bxor(hash, k)
    end

    hash = bit.bxor(hash, len)

    hash = bit.bxor(hash, bit.rshift(hash, 16))
    hash = hash * 0x85ebca6b
    hash = bit.bxor(hash, bit.rshift(hash, 13))
    hash = hash * 0xc2b2ae35
    hash = bit.bxor(hash, bit.rshift(hash, 16))

    return hash
end

-- Mersenne Twister implementation in Lua
local function init(rng, seed)
    rng.mt[0] = seed
    for i = 1, 623 do
        rng.mt[i] = bit.band((1812433253 * bit.bxor(rng.mt[i - 1], bit.rshift(rng.mt[i - 1], 30)) + i), 0xffffffff)
    end
    rng.index = 624
end

local function twist(rng)
    for i = 0, 623 do
        local y = bit.bor(bit.band(rng.mt[i], 0x80000000), bit.band(rng.mt[(i + 1) % 624], 0x7fffffff))
        rng.mt[i] = bit.bxor(rng.mt[(i + 397) % 624], bit.rshift(y, 1))
        if y % 2 ~= 0 then
            rng.mt[i] = bit.bxor(rng.mt[i], 0x9908b0df)
        end
    end
    rng.index = 0
end

local function extract_number(rng)
    if rng.index >= 624 then
        twist(rng)
    end

    local y = rng.mt[rng.index]
    y = bit.bxor(y, bit.rshift(y, 11))
    y = bit.bxor(y, bit.band(bit.lshift(y, 7), 0x9d2c5680))
    y = bit.bxor(y, bit.band(bit.lshift(y, 15), 0xefc60000))
    y = bit.bxor(y, bit.rshift(y, 18))

    rng.index = rng.index + 1
    return bit.band(y, 0xffffffff)
end

local function get_random_number(seed)
    local rng = { mt = {}, index = 0 }
    init(rng, seed)
    return extract_number(rng)
end
-- End of Mersenne Twister implementation

-- Util functions
-- ============================================================================
-- Internal function used to convert array of integers into Redis byte array
--
local function to_byte_array(json_str)    
    -- Initialize an empty table to hold the byte array
    local byte_array = {}    
    -- Convert each integer to an 8-byte representation and add to the byte array
    for _, num in ipairs(vector) do
        local byte_str = string.pack("I8", num)
        table.insert(byte_array, byte_str)
    end    
    
    -- Concatenate all the 8-byte strings into a single byte array
    local concatenated_bytes = table.concat(byte_array)
    
    -- Convert the byte array to a hexadecimal string
    local hex_str = {}
    for i = 1, #concatenated_bytes do
        table.insert(hex_str, string.format("%02X", string.byte(concatenated_bytes, i)))
    end
    
    return table.concat(hex_str)
end


local function count_trailing_zeros(num)
    local count = 0
    while num > 0 and bit.band(num, 1) == 0 do
        count = count + 1
        num = bit.rshift(num, 1)
    end
    return count
end

local function redis_rnd(token, p)
    local signed_number = murmurhash3_32(token, 0) -- Example function call
    local unsigned_number = bit.band(signed_number, 0xFFFFFFFF)
    
    -- Ensure the number is treated as unsigned
    if unsigned_number < 0 then
        unsigned_number = unsigned_number + 2^32
    end

    -- Other operations to generate first_p_bits and trailing_zeros
    local first_p_bits = bit.rshift(unsigned_number, 32 - p)
    local first_p_bits_integer = tonumber(first_p_bits) + 1
    local trailing_zeros = 0
    local temp = unsigned_number
    while temp > 0 and bit.band(temp, 1) == 0 do
        trailing_zeros = trailing_zeros + 1
        temp = bit.rshift(temp, 1)
    end

    return unsigned_number, first_p_bits_integer, trailing_zeros
end

-- Function to get a specified number of random keys
local function random_keys(num_keys)
    local random_keys = {}
    for i = 1, num_keys do
        local key = redis.call("RANDOMKEY")
        if key then
            table.insert(random_keys, key)
        end
    end
    return random_keys
end

-- End of Util function

-- Exported Functions
-- =============================================================================

-- Function updates '_last_modified_' field on each update of the hash
--
local function my_hset(keys, args)
    local hash = keys[1]
    local time = redis.call('TIME')[1]
    return redis.call('HSET', hash, '_last_modified_', time, unpack(args))
end

-- Function callable from outside Redis
--
local function redis_rand(keys, args)
    local token = args[1]
    local p = args[2]

    local random_number, first_p_bits, trailing_zeros = redis_rnd(token, p)

    -- Create a table with the return values
    local result = {
        random_number = random_number,
        first_p_bits = first_p_bits,
        trailing_zeros = trailing_zeros
    }

    -- Encode the table into a JSON string
    local json_result = cjson.encode(result)

    return json_result
end

-- Ingest only tokens creating token's hashes for each token;
-- keys = [token_key, entity_key]
-- args[1] is a precission of HllSet (number of leading bits used to number bins)
-- the reast of
-- args ia array of tokens to be processed
--
-- Fieldes token and refs are presented as sets
--
local function zero_num_to_digit(z_num)
    local n = tonumber(z_num)
    if n <= 1 then
        return 0
    end
    return 2^(n - 1)
end

local function create_zero_vector(size)
    local vector = {}
    -- Fill the table with zeros
    for i = 1, size do
        vector[i] = 0
    end
    return vector
end

local function update_vector(vector, index, value, operation)

    -- redis.log(redis.LOG_NOTICE, "vector: " .. tostring(vector))
    -- redis.log(redis.LOG_NOTICE, "index: " .. tostring(index))
    -- redis.log(redis.LOG_NOTICE, "value: " .. tostring(value))

    -- Check if the index is within the bounds of the vector
    if index < 1 then
        error("Index less than 1")
    elseif index > #vector then
        error("Index greatess than vector size: " .. #vector)
    end
    -- Update the element at the specified index
    if operation == "AND" then
        vector[index] = bit.band(vector[index], value)
    elseif operation == "OR" then
        vector[index] = bit.bor(vector[index], value)
    elseif operation == "XOR" then
        vector[index] = bit.bxor(vector[index], value)
    else
        return redis.error_reply("Unsupported operation")
    end
    -- Return the updated vector
    return vector
end

-- Performs bitwise operations on encoded as JSON strings vectors of integers
-- 
local function bit_ops(keys, args)
    local vector1 = cjson.decode(args[1])
    local vector2 = cjson.decode(args[2])
    local operation = args[3]

    local num_args = #args
    if num_args == 3 then
        redis.error_reply("Wrong number of arguments. Expected 3 arguments.")
    end

    local size_vector1 = #vector1
    local size_vector2 = #vector2

    if size_vector1 ~= size_vector2 then
        redis.error_reply("Vectors provided in args[1] and args[2] must be of the same size.")
    end

    local result = {}

    if operation == "AND" then
        for i = 1, #vector1 do
            table.insert(result, bit.band(vector1[i], vector2[i]))
        end
    elseif operation == "OR" then
        for i = 1, #vector1 do
            table.insert(result, bit.bor(vector1[i], vector2[i]))
        end
    elseif operation == "XOR" then
        for i = 1, #vector1 do
            table.insert(result, bit.bxor(vector1[i], vector2[i]))
        end
    else
        return redis.error_reply("Unsupported operation")
    end

    return cjson.encode(result)
end

-- ==============================================================================
-- Local functions to support Entity
-- ==============================================================================

-- Utility functions
-- ==============================================================================

local function bytes_to_int(byte_string)
    if type(byte_string) == "string" and #byte_string == 4 then
        local b1, b2, b3, b4 = byte_string:byte(1, 4)
        return b1 * 2^24 + b2 * 2^16 + b3 * 2^8 + b4
    else
        return 0
    end
end

local function int32_to_4bytes(num)
    local bytes = {}
    for i = 3, 0, -1 do
        bytes[#bytes + 1] = string.char(bit.band(bit.rshift(num, i * 8), 0xFF))
    end
    return table.concat(bytes)
end

local function json_vector_to_byte_string(json_vector)
    local vector = cjson.decode(json_vector)    
    -- Convert each integer to a 4-byte string and concatenate
    local byte_string = ""
    for _, int in ipairs(vector) do
        byte_string = byte_string .. int32_to_4bytes(int)
    end    
    return byte_string
end

local function byte_string_to_json_vector(byte_string)
    local vector = {}
    for i = 1, #byte_string, 4 do
        local byte_chunk = byte_string:sub(i, i + 3)
        local int = bytes_to_int(byte_chunk)
        table.insert(vector, i, int)
    end    
    return vector
end

-- ==============================================================================
-- Entity CRUD functions. All operations on Entity structure are performed in 
-- memory in Julia environment because Lua and Redis don't have adecvate support
-- ==============================================================================
-- Entity in Redis is a hash with following structure:
--  1. sha1::String
--  2. hll::byte_string from fixed size Vector{UInt32}
--  3. grad::Float64
--  4. op::nil or json string representing Operation{FuncType, ArgTypes}
-- ArgType is a tuple that lists sha1 Entities argument for given Entity

local function ingest_01(keys, args)
    local key1  = keys[1] -- prefix for token hash
    local key2  = keys[2] -- hash key for the Entity instance or graph node (they should be the same)
    local p     = tonumber(args[1])
    -- local batch = tonumber(args[2])
    local tokens = cjson.decode(args[2])
    local size = 2^p
    local counts = create_zero_vector(size)

    local success = true

    for i = 1, #tokens do
        local token = tokens[i]
        
        if #token > 2 then
            local random_number, first_p_bits, trailing_zeros = redis_rnd(token, p)

            local redis_key = key1 .. random_number
            local z_n = zero_num_to_digit(trailing_zeros)

            counts = update_vector(counts, first_p_bits, z_n, "OR")

            local hash_key = tostring("t:" .. redis_key)
            local skey = tostring(redis_key .. ":toks")
            local rkey = tostring(redis_key .. ":refs")
            redis.call("SADD", skey, token)
            redis.call("SADD", rkey, key2)
            local tokens = redis.call("SMEMBERS", skey)
            local refs = redis.call("SMEMBERS", rkey)
            local tokens_string = table.concat(tokens, ",")
            local refs_string = table.concat(refs, ",")
            local tf = redis.call("HGET", hash_key, "tf")
            -- redis.log(redis.LOG_NOTICE, "tf: " .. tostring(tf))
            if tf == nil or not tf then
                tf = 1
            else
                tf = tonumber(tf) + 1                
            end
            redis.call("HSET", hash_key, "id", random_number, "bin", first_p_bits, "zeros", trailing_zeros, "token", tokens_string, "refs", refs_string, "tf", tf)
        end
    end
    if success then
        local json_result = cjson.encode(counts)
        return json_result
    else
        return redis.error_reply("failure")
    end
end

local function store_entity(keys, args)
    local prefix    = keys[1]

    local sha1      = tostring(args[1])
    local card      = tonumber(args[2])
    local dataset   = json_vector_to_byte_string(args[3])
    local grad      = tonumber(args[4])
    local op_op     = tostring(args[5])
    local op_args   = cjson.decode(args[6])

    local hash_key  = tostring(prefix .. ":" .. sha1)
    -- local data_str  = json_vector_to_byte_string(dataset)
    local op_argstr = table.concat(op_args, ",")

    redis.call("HSET", hash_key, "id", sha1, "card", card, "dataset", dataset, "op_op", op_op, "op_args", op_argstr)

    return "OK"
end

local function retrieve_entity(keys, args)
    local prefix    = keys[1]
    local sha1      = tostring(args[1])
    local hash_key  = tostring(prefix .. ":" .. sha1)

    return redis.call("HGETALL", hash_key)    
end

-- ===============================================================================
-- Scanning and retrieving tokens
-- ===============================================================================
-- Function to check if a specific reference is in the refs field
local function ref_in_refs(refs, ref)
    for token in string.gmatch(refs, '([^,]+)') do
        if token == ref then
            return true
        end
    end
    return false
end

-- Function to convert a space-separated string into a Lua table
local function split_string_to_table(str)
    local t = {}
    for word in string.gmatch(str, "%S+") do
        table.insert(t, word)
    end
    return t
end

-- Function to check if there is a non-empty intersection between refs and specific_refs
local function has_intersection(refs, specific_refs)
    local refs_set = {}
    local t = split_string_to_table(specific_refs)
    for token in string.gmatch(refs, '([^,]+)') do
        refs_set[token] = true
    end
    for _, ref in ipairs(t) do
        if refs_set[ref] then
            return true
        end
    end
    return false
end

-- Function to evaluate a condition
local function evaluate_condition(value, operator, compare_value)
    if operator == "==" then
        return value == compare_value
    elseif operator == ">" then
        return tonumber(value) > tonumber(compare_value)
    elseif operator == "<" then
        return tonumber(value) < tonumber(compare_value)
    elseif operator == "!=" then
        return value ~= compare_value
    elseif operator == "has" then
        return ref_in_refs(value, compare_value)
    elseif operator == "match" then
        return has_intersection(value, compare_value)
    else
        return false
    end
end

-- Function to check if a hash satisfies all conditions
local function satisfies_conditions(key, conditions)
    for i = 1, #conditions, 3 do
        local field = conditions[i]
        local operator = conditions[i + 1]
        local compare_value = conditions[i + 2]
        local value = redis.call("HGET", key, field)
        if not value or not evaluate_condition(value, operator, compare_value) then
            return false
        end
    end
    return true
end

local function retrieve_tokens(keys, args)
    local prefix = keys[1]
    local conditions = args

    local cursor = "0"
    local matching_keys = {}

    repeat
        local result = redis.call("SCAN", cursor, "MATCH", prefix .. "*")
        cursor = result[1]
        local keys = result[2]

        for _, key in ipairs(keys) do
            if satisfies_conditions(key, conditions) then
                -- table.insert(matching_keys, key)
                local tf_value = redis.call("HGET", key, "tf")
                table.insert(matching_keys, {key, tf_value})
            end
        end
    until cursor == "0"

    return matching_keys
end

-- ===============================================================================
-- Function Registration
-- ===============================================================================

-- redis.register_function('UPDATETOKS', update_toks)
redis.register_function('my_hset',      my_hset)
redis.register_function('bit_ops',      bit_ops)
redis.register_function('redis_rand',   redis_rand)
redis.register_function('ingest_01',    ingest_01)
redis.register_function('store_entity', store_entity)
redis.register_function('retrieve_entity', retrieve_entity)
redis.register_function('retrieve_tokens', retrieve_tokens)