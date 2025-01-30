include("sets32.jl")
include("utils.jl")

module Entity

    export Instance

    using ..HllSets

    using JSON3
    using PyCall
    # Default Redis instance
    redis = pyimport("redis")
    r = redis.Redis(host="localhost", port=6379)

    # Base functions that we're going to be overriding 
    # import Base: ==, show, union, intersect, xor, comp, copy, negation, diff, adv

    # Operation
    struct Operation{FuncType}
        op::FuncType
        args::Vector
    end

    function op_op(op::Union{Operation, Nothing})
        if op == nothing
            return "nothing"
        else
            return string(op.op)
        end
    end

    function op_args(op::Union{Operation, Nothing}) 
        if op == nothing
            return ["nothing"]
        else
            return [inst.sha1 for inst in op.args]
        end
    end

    function store_entity(r::PyObject, card::Int, hll::HllSets.HllSet{P}; grad=0.0, op=nothing,  prefix::String="b") where P
        sha1 = string(HllSets.id(hll))
        dataset = JSON3.write(HllSets.dump(hll))
        args = JSON3.write(op_args(op))
        # println(args)
        return r.fcall("store_entity", 1, prefix, sha1, card, dataset, 0, op_op(op), args), sha1
    end

    function store_entity(r::PyObject, card::Int, dataset::String, sha1::String; grad=0.0, op=nothing,  prefix::String="b") 
        args = JSON3.write(op_args(op))
        # println(args)
        return r.fcall("store_entity", 1, prefix, sha1, card, dataset, 0, op_op(op), args), sha1
    end

    function parse_entity_output(output::Vector)
        # Initialize an empty dictionary
        result = Dict{String, Any}() 
        println(output)       
        # Iterate through the vector and extract key-value pairs
        i = 1
        while i <= length(output)
            key = output[i]
            value = output[i + 1] 
            println(key, ": ", value)           
            if key == "dataset"
                # Convert the dataset string to an array of integers
                dataset_bytes = collect(UInt8, value)
                dataset_ints = reinterpret(Int32, dataset_bytes)
                result[key] = dataset_ints
            # elseif key == "op"
            else
                result[key] = value
            end            
            i += 2
        end        
        return result
    end

    function retrieve_entity(sha1::String; r::PyObject=r,  prefix::String="b")
        output = r.fcall("retrieve_entity", 1, prefix, sha1)
        json = parse_entity_output(output)
    end

    # local prefix    = keys[1]
    # local sha1      = tostring(args[1])
    # local dataset   = cjson.decode(args[2])
    # local grad      = tonumber(args[3])
    # local op_op     = tostring(args[4])
    # local op_args   = cjson.decode(args[5])

    mutable struct Instance{P}
        sha1::String
        card::Int
        hll::HllSets.HllSet{P}
        grad::Float64
        op::Union{Operation{FuncType}, Nothing} where {FuncType}

        # Constructor with keyword arguments
        # By default new instance is created in buffer (prefix = "b")
        function Instance{P}(r::PyObject, hll::HllSets.HllSet{P}; grad=0.0, op=nothing, prefix::String="b") where {P}
            card = HllSets.count(hll)
            response, sha1 = store_entity(r, card, hll, grad=grad, op=op, prefix=prefix)
            new{P}(sha1, card, hll, grad, op)
        end

        function Instance{P}(hll::HllSets.HllSet{P}; grad=0.0, op=nothing, prefix::String="b") where {P}
            # response, sha1 = store_entity(r, hll, grad=grad, op=op, prefix=prefix)
            card = HllSets.count(hll)
            sha1 = HllSets.id(hll)
            new{P}(sha1, card, hll, grad, op)
        end
    end

    function Base.show(io::IO, instance::Instance)
        sha1_str = instance.sha1 === nothing ? "nothing" : string(instance.sha1)
        card = instance.card === nothing ? "nothing" : string(instance.card)
        hll_str = instance.hll === nothing ? "nothing" : string(instance.hll)
        grad_str = instance.grad === nothing ? 0.0 : instance.grad
        op_str = instance.op === nothing ? "nothing" : string(instance.op)
        
        println(io, "\nInstance(\n sha1: ", sha1_str,
            "\n card: ", card, "\n hll: ", hll_str, " grad: ", grad_str, "\n op: ", op_str, ")\n")
    end

    function isequal(a::Instance, b::Instance)
        return HllSets.isequal(a.sha1, b.sha1) && a.grad == b.grad
    end

    #------------------------------------------------------------
    # Set of Instance operations to support Static Entity Structure
    #------------------------------------------------------------
    function copy(r::PyObject, a::Instance{P}) where {P}
        return Instance{P}(r, HllSets.copy!(a.hll); grad=a.grad, op=a.op)
    end

    # negation
    function negation(r::PyObject, a::Instance{P}) where {P}
        return Instance{P}(r, HllSets.copy!(a.hll); grad=-a.grad, op=a.op)
    end
    # union
    function union(r::PyObject, a::Instance{P}, b::Instance{P}) where {P}
        hll_result = HllSets.union(a.hll, b.hll)
        op_result = Operation(union, [a, b])
        return Instance{P}(r, hll_result; grad=0.0, op=op_result)
    end

    # union backprop
    function backprop!(instance::Instance{P}, 
            instance_op::Union{Operation{FuncType}, Nothing}=instance.op) where {P, FuncType<:typeof(union)}
        if (instance.op != nothing) && (instance.op === instance_op) && (instance_op.op === union)
            instance_op.args[1].grad += instance.grad
            instance_op.args[2].grad += instance.grad
        else
            println("Error: Operation not supported for terminal node")
        end
    end

    # intersect - intersection
    function intersect(r::PyObject, a::Instance{P}, b::Instance{P}) where {P}
        hll_result = HllSets.intersect(a.hll, b.hll)
        op_result = Operation(intersect, [a, b])
        return Instance{P}(r, hll_result; grad=0.0, op=op_result)
    end

    # intersect backprop
    function backprop!(instance::Instance{P}, instance_op::Operation{FuncType}) where {P, FuncType<:typeof(intersect)}
        if (instance.op != nothing) && (instance.op === instance_op) && (instance_op.op === intersect)
            instance_op.args[1].grad += instance.grad
            instance_op.args[2].grad += instance.grad
        else
            println("Error: Operation not supported for terminal node")
        end
    end

    # xor 
    function xor(r::PyObject, a::Instance{P}, b::Instance{P}) where {P}
        hll_result = HllSets.set_xor(a.hll, b.hll)
        op_result = Operation(xor, [a, b])
        return Instance{P}(r, hll_result; grad=0.0, op=op_result)
    end

    # xor backprop
    function backprop!(instance::Instance{P}, instance_op::Operation{FuncType}) where {P, FuncType<:typeof(xor)}
        if (instance.op != nothing) && (instance.op === instance_op) && (instance_op.op === xor)
            instance_op.args[1].grad += instance.grad
            instance_op.args[2].grad += instance.grad
        else
            println("Error: Operation not supported for terminal node")
        end
    end

    # comp - complement returns the elements that are in the a set but not in the b
    function comp(a::Instance{P}, b::Instance{P}; opType=comp) where {P}
        # b should not be empty
        HllSets.count(b.hll) > 0  || throw(ArgumentError("HllSet{P} cannot be empty"))

        hll_result = HllSets.set_comp(a.hll, b.hll)
        op_result = Operation(comp, [a, b])
        comp_grad = HllSets.count(hll_result) / HllSets.count(a.hll)
        println("comp_grad: ", comp_grad)

        return Instance{P}(hll_result; grad=comp_grad, op=op_result)
    end

    # comp backprop
    # Technically this operation is changing first argument, so, it's not exactly a static operation.
    # We are keeping it here because it's not dynamic operation ether bur we are updating grad for the first argument.
    function backprop!(instance::Instance{P}, instance_op::Operation{FuncType}) where {P, FuncType<:typeof(comp)}
        if (Instance.op != nothing) && (instance.op === instance_op) && (instance_op.op === comp)
            instance_op.args[1].grad += instance.grad
            # instance_op.args[2].grad += instance.grad
        else
            println("Error: Operation not supported for terminal node")
        end
    end

    #------------------------------------------------------------
    # Set of Instance operations to support Dynamic Entity Structure
    #------------------------------------------------------------
    """
        This convenience methods that semantically reflect the purpose of using set_comp function
        in case of comparing two states of the same set now (current) and as it was before (previous).
        - set_added - returns the elements that are in the current set but not in the previous
        - set_deleted - returns the elements that are in the previous set but not in the current
    """
    function added(r::PyObject, current::Instance{P}, previous::Instance{P}) where {P} 
        length(previous.hll.counts) == length(current.hll.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        
        result = comp(previous, current)
        println(result)
        op_result = Operation(added, [current, previous])
        added_grad = result.grad

        return Instance{P}(r, result.hll; grad=added_grad, op=op_result)
    end

    # added backprop
    function backprop!(instance::Instance{P}, instance_op::Operation{FuncType}) where {P, FuncType<:typeof(added)}
        if (instance.op != nothing) && (instance.op === instance_op) && (instance_op.op === added)
            # instance_op.args[1].grad *= instance.grad
            instance_op.args[2].grad *= instance.grad
        else
            println("Error: Operation not supported for terminal node")
        end
    end

    function deleted(r::PyObject, current::Instance{P}, previous::Instance{P}) where {P} 
        length(previous.hll.counts) == length(current.hll.counts) || throw(ArgumentError("HllSet{P} must have same size"))

        result = comp(current, previous)
        op_result = Operation(deleted, [current, previous])
        deleted_grad = result.grad

        return Instance{P}(r, result.hll; grad=deleted_grad, op=op_result)
    end

    # deleted backprop
    function backprop!(instance::Instance{P}, instance_op::Operation{FuncType}) where {P, FuncType<:typeof(deleted)}
        if (instance.op != nothing) && (instance.op === instance_op) && (instance_op.op === deleted)
            instance_op.args[1].grad *= instance.grad
            # instance_op.args[2].grad *= instance.grad
        else
            println("Error: Operation not supported for terminal node")
        end
    end

    function retained(r::PyObject, current::Instance{P}, previous::Instance{P}) where {P} 
        length(previous.hll.counts) == length(current.hll.counts) || throw(ArgumentError("HllSet{P} must have same size"))
        
        hll_result = HllSets.intersect(current.hll, previous.hll)
        op_result = Operation(retained, [current, previous])
        retained_grad = HllSets.count(hll_result) / HllSets.count(HllSets.union(current.hll, previous.hll))

        return Instance{P}(r, hll_result; grad=retained_grad, op=op_result)
    end

    # retained backprop
    function backprop!(instance::Instance{P}, instance_op::Operation{FuncType}) where {P, FuncType<:typeof(retained)}
        if (instance.op != nothing) && (instance.op === instance_op) && (instance_op.op === retained)
            instance_op.args[1].grad *= instance.grad
            instance_op.args[2].grad *= instance.grad
        else
            println("Error: Operation not supported for terminal node")
        end
    end

    # difference - diff 
    function diff(r::PyObject, a::Instance{P}, b::Instance{P}) where {P}
        d = deleted(r, a, b)
        rt = retained(r, a, b)
        n = added(r, a, b)
        return d, rt, n
    end

    # advance - Allows us to calculate the gradient for the advance operation
    # We are using 'advance' name to reflect the transformation of the set 
    # from the previous state to the current state
    function advance(r::PyObject, a::Instance{P}, b::Instance{P}) where {P}
        d, rt, n = diff(r, a, b)
        hll_res = HllSets.union(n.hll, rt.hll)
        op_result = Operation(advance, [d, rt, n])
        # calculate the gradient for the advance operation as 
        # the difference between the number of elements in the n set 
        # and the number of elements in the d set

        grad_res = HllSets.count(a.hll) / HllSets.count(b.hll)  # This is the simplest way to calculate the gradient
        
        # Create updated version of the Instance
        return Instance{P}(r, hll_res; grad=grad_res, op=op_result)
    end

    """ 
        This version of advance operation generates new unknown set from the current set
        that we are using as previous set. 
        Instance b has some useful information about current state of the set:
            - b.hll - current state of the set
            - b.grad - gradient value that we are going to use to calculate the gradient for the advance operation
            - b.op - operation that we are going to use to calculate the gradient for the advance operation. 
                    op has information about how we got to the current set b.
                    - op.args[1] - deleted set
                    - op.args[2] - retained set
                    - op.args[3] - added set
        We are going to use this information to construct the new set that represents the unknown state of the set.
    """
    function advance(r::PyObject, ::Colon; b::Instance{P}) where {P}
        # Create a new empty set
        a = HllSets.create(b.hll)
        d, rt, n = diff(r, a, b)
        op_result = Operation(advance, [d, rt, n])
        # calculate the gradient for the advance operation as 
        # the number of elements in the a set
        grad_res = HllSets.count(a.hll)  # This is the simplest way to calculate the gradient
        
        # Create updated version of the instance
        return Instance{P}(r, hll_res; grad=grad_res, op=op_result)
    end

    function backprop!(instance::Instance{P}, instance_op::Operation{FuncType}) where {P, FuncType<:typeof(advance)}
        if (instance.op != nothing) && (instance.op === instance_op) && (instance_op.op === advance)
            if instance_op.args[1].op !== nothing
                instance_op.args[1].grad *= instance.grad
            end
            if instance_op.args[2].op !== nothing                
                instance_op.args[2].grad *= instance.grad
            end
            if instance_op.args[3].op !== nothing
                instance_op.args[3].grad *= instance.grad
            end
        else
            println("Error: Operation not supported for terminal node")
        end
    end
end # module SetGrad