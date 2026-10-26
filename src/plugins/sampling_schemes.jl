#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct InSampleMonteCarlo <: AbstractSamplingScheme
    max_depth::Int
    rollout_limit::Function
    initial_node::Union{Nothing,Int}
end

function InSampleMonteCarlo(;
    max_depth::Int = 0,
    rollout_limit::Function = i -> typemax(Int),
    initial_node::Union{Nothing,Int} = nothing,
)
    new_rollout = let i = 0
        () -> (i += 1; rollout_limit(i))
    end
    return InSampleMonteCarlo(max_depth, new_rollout, initial_node)
end

function sample_noise(noise_terms::Vector{<:Noise})
    if length(noise_terms) == 0
        return nothing
    end
    cumulative_probability = sum(noise.probability for noise in noise_terms)
    if cumulative_probability > 1.0 + 1e-6
        error("Cumulative probability cannot be greater than 1.0.")
    end
    rnd = rand() * cumulative_probability
    for noise in noise_terms
        rnd -= noise.probability
        if rnd <= 0.0
            return noise.term
        end
    end
    return error(
        "Internal MSUC error: unable to sample noise from $(noise_terms)",
    )
end

function sample_scenario(graph::PolicyGraph, sampling_scheme::InSampleMonteCarlo)
    max_depth = min(sampling_scheme.max_depth, sampling_scheme.rollout_limit())
    scenario_path = Tuple{Int,Any}[]
    node_index = something(sampling_scheme.initial_node, graph.first_node)
    while true
        node = graph[node_index]
        push!(scenario_path, (node_index, sample_noise(node.noise_terms)))
        if node.next_node === nothing ||
           0 < max_depth <= length(scenario_path)
            return scenario_path, false
        end
        node_index = node.next_node
    end
end

mutable struct PSRSamplingScheme{A} <: AbstractSamplingScheme
    N::Int
    sampling_scheme::A
    scenarios::Vector{Any}
    counter::Int
    lock::ReentrantLock

    function PSRSamplingScheme(
        N::Int;
        sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
    )
        lock = ReentrantLock()
        return new{typeof(sampling_scheme)}(N, sampling_scheme, Any[], 0, lock)
    end
end

function Base.show(io::IO, h::PSRSamplingScheme)
    print(io, "A sampler with $(length(h.scenarios)) scenarios like PSR does.")
    return
end

function sample_scenario(
    graph::PolicyGraph,
    s::PSRSamplingScheme{A};
    kwargs...,
) where {A}
    lock(s.lock)
    try
        s.counter += 1
        if s.counter > s.N
            s.counter = 1
        end
        if s.counter > length(s.scenarios)
            push!(
                s.scenarios,
                sample_scenario(graph, s.sampling_scheme; kwargs...),
            )
        end
        return s.scenarios[s.counter]
    finally
        unlock(s.lock)
    end
end
