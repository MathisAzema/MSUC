#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# ========================= Monte Carlo Sampling Scheme ====================== #

struct InSampleMonteCarlo <: AbstractSamplingScheme
    max_depth::Int
    terminate_on_cycle::Bool
    terminate_on_dummy_leaf::Bool
    rollout_limit::Function
    initial_node::Any
end

"""
    InSampleMonteCarlo(;
        max_depth::Int = 0,
        terminate_on_cycle::Function = false,
        terminate_on_dummy_leaf::Function = true,
        rollout_limit::Function = (i::Int) -> typemax(Int),
        initial_node::Any = nothing,
    )

A Monte Carlo sampling scheme using the in-sample data from the policy graph
definition.

If `terminate_on_cycle`, terminate the forward pass once a cycle is detected.
If `max_depth > 0`, return once `max_depth` nodes have been sampled.
If `terminate_on_dummy_leaf`, terminate the forward pass with 1 - probability of
sampling a child node.

Note that if `terminate_on_cycle = false` and `terminate_on_dummy_leaf = false`
then `max_depth` must be set > 0.

Control which node the trajectories start from using `initial_node`. If it is
left as `nothing`, the root node is used as the starting node.

You can use `rollout_limit` to set iteration specific depth limits. For example:

    InSampleMonteCarlo(rollout_limit = i -> 2 * i)
"""
function InSampleMonteCarlo(;
    max_depth::Int = 0,
    terminate_on_cycle::Bool = false,
    terminate_on_dummy_leaf::Bool = true,
    rollout_limit::Function = i -> typemax(Int),
    initial_node::Any = nothing,
)
    if !terminate_on_cycle && !terminate_on_dummy_leaf && max_depth == 0
        error(
            "terminate_on_cycle and terminate_on_dummy_leaf cannot both be " *
            "false when max_depth=0.",
        )
    end
    new_rollout = let i = 0
        () -> (i += 1; rollout_limit(i))
    end
    return InSampleMonteCarlo(
        max_depth,
        terminate_on_cycle,
        terminate_on_dummy_leaf,
        new_rollout,
        initial_node,
    )
end

function get_noise_terms(
    sampling_scheme::InSampleMonteCarlo,
    node::Node{T},
    node_index::T,
) where {T}
    return node.noise_terms
end

function get_children(
    sampling_scheme::InSampleMonteCarlo,
    node::Node{T},
    node_index::T,
) where {T}
    return node.children
end

function get_root_children(
    sampling_scheme::InSampleMonteCarlo,
    graph::PolicyGraph{T},
) where {T}
    return graph.root_children
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

function sample_scenario(
    graph::PolicyGraph{T},
    sampling_scheme::InSampleMonteCarlo,
) where {T}
    max_depth = min(sampling_scheme.max_depth, sampling_scheme.rollout_limit())
    # Storage for our scenario. Each tuple is (node_index, noise.term).
    scenario_path = Tuple{T,Any}[]
    # We only use visited_nodes if terminate_on_cycle=true. Just initialize
    # anyway.
    visited_nodes = Set{T}()
    # Begin by sampling a node from the children of the root node.
    node_index = something(
        sampling_scheme.initial_node,
        sample_noise(get_root_children(sampling_scheme, graph)),
    )::T
    while true
        node = graph[node_index]
        noise_terms = get_noise_terms(sampling_scheme, node, node_index)
        children = get_children(sampling_scheme, node, node_index)
        noise = sample_noise(noise_terms)
        push!(scenario_path, (node_index, noise))
        # Termination conditions:
        if length(children) == 0
            # 1. Our node has no children, i.e., we are at a leaf node.
            return scenario_path, false
        elseif sampling_scheme.terminate_on_cycle && node_index in visited_nodes
            # 2. terminate_on_cycle = true and we have detected a cycle.
            return scenario_path, true
        elseif 0 < sampling_scheme.max_depth <= length(scenario_path)
            # 3. max_depth > 0 and we have explored max_depth number of nodes.
            return scenario_path, false
        elseif sampling_scheme.terminate_on_dummy_leaf &&
               rand() < 1 - sum(child.probability for child in children)
            # 4. we sample a "dummy" leaf node in the next step due to the
            # probability of the child nodes summing to less than one.
            return scenario_path, false
        end
        # We only need to store a list of visited nodes if we want to terminate
        # due to the presence of a cycle.
        if sampling_scheme.terminate_on_cycle
            push!(visited_nodes, node_index)
        end
        # Sample a new node to transition to.
        node_index = sample_noise(children)::T
    end
    # Throw an error because we should never end up here.
    return error(
        "Internal MSUC error: something went wrong sampling a scenario.",
    )
end

"""
    PSRSamplingScheme(N::Int; sampling_scheme = InSampleMonteCarlo())

A sampling scheme with `N` scenarios, similar to how PSR does it.
"""
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
    graph::PolicyGraph{T},
    s::PSRSamplingScheme{A};
    kwargs...,
) where {T,A}
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

"""
    SimulatorSamplingScheme(simulator::Function)

Create a sampling scheme based on a univariate scenario generator `simulator`,
which returns a `Vector{Float64}` when called with no arguments like
`simulator()`.

This sampling scheme must be used with a Markovian graph constructed from the
same `simulator`.

The sample space for [`MSUC.parameterize`](@ref) must be a tuple with 1 or 2
values, value is the Markov state and the second value is the random variable
for the current node. If the node is deterministic, use `Ω = [(markov_state,)]`.

This sampling scheme generates a new scenario by calling `simulator()`, and then
picking the sequence of nodes in the Markovian graph that is closest to the new
trajectory.

## Example

```julia
julia> using MSUC

julia> import HiGHS

julia> simulator() = cumsum(rand(10))
simulator (generic function with 1 method)

julia> model = MSUC.PolicyGraph(
           MSUC.MarkovianGraph(simulator; budget = 20, scenarios = 100);
           sense = :Max,
           upper_bound = 12,
           optimizer = HiGHS.Optimizer,
       ) do sp, node
           t, markov_state = node
           @variable(sp, x >= 0, MSUC.State, initial_value = 1)
           @variable(sp, u >= 0)
           @constraint(sp, x.out == x.in - u)
           # Elements of Ω MUST be a tuple in which `markov_state` is the first
           # element.
           Ω = [(markov_state, (u = u_max,)) for u_max in (0.0, 0.5)]
           MSUC.parameterize(sp, Ω) do (markov_state, ω)
               set_upper_bound(u, ω.u)
               @stageobjective(sp, markov_state * u)
           end
       end;

julia> MSUC.train(
           model;
           print_level = 0,
           iteration_limit = 10,
           sampling_scheme = MSUC.SimulatorSamplingScheme(simulator),
       )

```
"""
mutable struct SimulatorSamplingScheme{F} <: AbstractSamplingScheme
    simulator::F
end

function Base.show(io::IO, h::SimulatorSamplingScheme)
    print(io, "SimulatorSamplingScheme")
    return
end

function _closest_index(graph, t, value)
    min_value, min_dist = value, Inf
    for (t_, value_) in keys(graph.nodes)
        if t_ == t
            if abs(value - value_) < min_dist
                min_value = value_
                min_dist = abs(value - value_)
            end
        end
    end
    return (t, min_value)
end

function sample_scenario(
    graph::PolicyGraph{Tuple{Int,Float64}},
    s::SimulatorSamplingScheme{F},
) where {F}
    scenario_path = Tuple{Tuple{Int,Float64},Any}[]
    for (t, value) in enumerate(s.simulator())
        node_index = _closest_index(graph, t, value)
        node = graph[node_index]
        noise_terms = get_noise_terms(InSampleMonteCarlo(), node, node_index)
        noise = sample_noise(noise_terms)
        @assert noise[1] == node_index[2]
        ω = length(noise) == 1 ? (value,) : (value, noise[2])
        push!(scenario_path, (node_index, ω))
    end
    return scenario_path, false
end
