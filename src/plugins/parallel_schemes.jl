#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function _should_log(options, log_frequency::Int)
    return options.print_level > 0 &&
           mod(length(options.log), log_frequency) == 0
end

function _should_log(options, log_frequency::Function)
    return options.print_level > 0 && log_frequency(options.log)
end

function log_iteration(options; force_if_needed::Bool = false)
    force_if_needed &= options.last_log_iteration[] != length(options.log)
    if force_if_needed || _should_log(options, options.log_frequency)
        print_helper(print_iteration, options.log_file_handle, options.log[end])
        flush(options.log_file_handle)
        options.last_log_iteration[] = length(options.log)
    end
    return
end

struct Serial <: AbstractParallelScheme end

Base.show(io::IO, ::Serial) = print(io, "serial mode")

interrupt(::Serial) = nothing

function master_loop(
    ::Serial,
    model::PolicyGraph,
    options::Options,
)
    _initialize_solver(model; throw_error = false)
    status = nothing
    while status === nothing
        status = disable_sigint() do
            result = iteration(model, options)
            options.post_iteration_callback(result)
            log_iteration(options)
            if result.has_converged
                return result.status
            end
            return nothing
        end
    end
    return status
end

function _simulate(
    model::PolicyGraph,
    ::Serial,
    number_replications::Int,
    variables::Vector{Symbol};
    kwargs...,
)
    _initialize_solver(model; throw_error = false)
    return map(
        i -> _simulate(model, variables; kwargs...),
        1:number_replications,
    )
end

struct Threaded <: AbstractParallelScheme end

Base.show(io::IO, ::Threaded) = print(io, "Threaded()")

interrupt(::Threaded) = nothing

function master_loop(
    ::Threaded,
    model::PolicyGraph,
    options::Options,
)
    max_threads = Threads.nthreads()
    num_nodes = length(model.nodes)
    if num_nodes < max_threads
        @warn(
            "There are fewer nodes in the graph ($num_nodes) than there are " *
            "threads available ($max_threads). Limiting the number of " *
            "threads to $num_nodes."
        )
        max_threads = num_nodes
    end
    _initialize_solver(model; throw_error = false)
    keep_iterating, status = true, nothing
    @sync for _ in 1:max_threads
        Threads.@spawn begin
            try
                while keep_iterating
                    disable_sigint() do
                        result = iteration(model, options)
                        lock(options.lock) do
                            options.post_iteration_callback(result)
                            log_iteration(options)
                            if result.has_converged
                                keep_iterating = false
                                status = result.status
                            end
                            return
                        end
                        return
                    end
                end
            finally
                lock(options.lock) do
                    keep_iterating = false
                    return
                end
            end
        end
    end
    return status
end

function _chunk_split(num_items, num_chunks)
    w = div(num_items, num_chunks)
    chunks = UnitRange{Int}[]
    offset = 0
    for i in 1:num_chunks-1
        push!(chunks, (offset+1):(offset+w))
        offset += w
    end
    push!(chunks, (offset+1):num_items)
    return chunks
end

function _simulate(
    model::PolicyGraph,
    ::Threaded,
    number_replications::Int,
    variables::Vector{Symbol};
    kwargs...,
)
    max_threads = Threads.nthreads()
    num_nodes = length(model.nodes)
    if num_nodes < max_threads
        @warn(
            "There are fewer nodes in the graph ($num_nodes) than there are " *
            "threads available ($max_threads). Limiting the number of " *
            "threads to $num_nodes."
        )
        max_threads = num_nodes
    end
    _initialize_solver(model; throw_error = false)
    ret = Vector{Vector{Dict{Symbol,Any}}}(undef, number_replications)
    @sync for chunk in _chunk_split(number_replications, max_threads)
        Threads.@spawn for j in chunk
            ret[j] = _simulate(model, variables; kwargs...)
        end
    end
    return ret
end
