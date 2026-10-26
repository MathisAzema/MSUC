#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct Graph
    stages::Int
end


sort_nodes(nodes) = sort!(nodes)

function Base.show(io::IO, graph::Graph)
    println(io, "Root\n 0\nNodes")
    for t in 1:graph.stages
        println(io, " ", t)
    end
    print(io, "Arcs")
    for t in 0:(graph.stages-1)
        print(io, "\n ", t, " => ", t + 1)
    end
    return
end

struct Noise{T}
    term::T
    probability::Float64
end

struct State{T}
    in::T
    out::T
end

mutable struct IntervalCut
    alpha::Float64
    tau::Dict{Tuple{Int64,Int64,Int64},Float64}
    beta::Dict{Tuple{Int64,Int64,Int64},Float64}
    delta::Vector{Float64}
    constraint_ref::JuMP.ConstraintRef
end

mutable struct Node
    index::Int
    subproblem::JuMP.Model
    next_node::Union{Nothing,Int}
    noise_terms::Vector{Noise}
    parameterize::Function
    states::Dict{Symbol,State{JuMP.VariableRef}}
    states_power::Dict{Symbol,State{JuMP.VariableRef}}
    states_gamma::Dict{Tuple{Int,Int,Int},JuMP.VariableRef}
    states_zeta::Vector{State{JuMP.VariableRef}}
    stage_objective::Any
    stage_objective_set::Bool
    bellman_function::Any
    has_integrality::Bool
    optimizer::Any
    ext::Dict{Symbol,Any}
    lock::ReentrantLock
    incoming_state_bounds::Dict{Symbol,Tuple{Float64,Float64,Bool}}
    interval_cuts::Vector{IntervalCut}
    demand_constraint::Vector{JuMP.ConstraintRef}
    computation_time::Vector{Float64}
end

function Base.show(io::IO, node::Node)
    println(io, "Node $(node.index)")
    println(io, "  # State variables : ", length(node.states))
    println(io, "  # Next node       : ", something(node.next_node, "none"))
    println(io, "  # Noise terms     : ", length(node.noise_terms))
    return
end

struct Log
    iteration::Int
    bound::Float64
    simulation_value::Float64
    time::Float64
    pid::Int
    total_solves::Int
    duality_key::String
    serious_numerical_issue::Bool
end

struct TrainingResults
    status::Symbol
    log::Vector{Log}
end

struct Method
    string::String
end

ExtendedFormulation = Method("ExtendedFormulation")
SDDPFormulation = Method("SDDP")

mutable struct PolicyGraph
    objective_sense::MOI.OptimizationSense
    root_node::Int
    first_node::Int
    initial_root_state::Dict{Symbol,Float64}
    nodes::Dict{Int,Node}
    most_recent_training_results::Union{Nothing,TrainingResults}
    ext::Dict{Symbol,Any}
    timer_output::TimerOutputs.TimerOutput
    lock::ReentrantLock

    instance::Instance
    method::Method
    force::Float64

    function PolicyGraph(instance::Instance, method::Method, force::Float64)
        return new(
            MOI.MIN_SENSE,
            0,
            1,
            Dict{Symbol,Float64}(),
            Dict{Int,Node}(),
            nothing,
            Dict{Symbol,Any}(),
            TimerOutputs.TimerOutput(),
            ReentrantLock(),
            instance,
            method,
            force,
        )
    end
end

function Base.show(io::IO, graph::PolicyGraph)
    N = length(graph.nodes)
    println(io, "A policy graph with $(N) nodes.")
    nodes = sort_nodes(collect(keys(graph.nodes)))
    if N < 10
        println(io, " Node indices: ", join(nodes, ", "))
    else
        println(io, " Node indices: ", nodes[1], ", ..., ", nodes[end])
    end
    return
end

function Base.getindex(graph::PolicyGraph, index::Int)
    return graph.nodes[index]
end

function PolicyGraph(
    subproblem_builder::Function,
    instance::Instance,
    force::Float64,
    graph::Graph,
    method::Method;
    lower_bound = -Inf,
    optimizer = nothing,
)
    if lower_bound === -Inf
        error(
            "You must specify a finite lower bound on the objective value" *
            " using the `lower_bound = value` keyword argument.",
        )
    end
    bellman_function = BellmanFunction(; lower_bound = lower_bound)
    policy_graph = PolicyGraph(instance, method, force)
    for node_index in 1:graph.stages
        subproblem = JuMP.Model()
        node = Node(
            node_index,
            subproblem,
            node_index < graph.stages ? node_index + 1 : nothing,
            Noise[],
            (ω) -> nothing,
            Dict{Symbol,State{JuMP.VariableRef}}(),
            Dict{Symbol,State{JuMP.VariableRef}}(),
            Dict{Tuple{Int,Int,Int},JuMP.VariableRef}(),
            State{JuMP.VariableRef}[],
            0.0,
            false,
            nothing,
            false,
            optimizer,
            Dict{Symbol,Any}(),
            ReentrantLock(),
            Dict{Symbol,Tuple{Float64,Float64,Bool}}(),
            IntervalCut[],
            JuMP.ConstraintRef[],
            Float64[],
        )
        subproblem.ext[:MSUC_policy_graph] = policy_graph
        policy_graph.nodes[node_index] = subproblem.ext[:MSUC_node] = node
        JuMP.set_objective_sense(subproblem, policy_graph.objective_sense)

        if method == ExtendedFormulation
            constraint, gamma_var = subproblem_builder(instance, force, subproblem, node_index)

            node.demand_constraint = constraint
            node.states_gamma = gamma_var
        else
            subproblem_builder(instance, force, subproblem, node_index)
        end
        for (state_sym, state_var) in node.states
            if occursin("power", String(state_sym))
                node.states_power[state_sym] = state_var
            end
        end
        node.states_zeta = [
            node.states[Symbol("ζ[$i]")] for
            i in 1:MSUC.zeta_dimension(instance.arma)
        ]

        if length(node.noise_terms) == 0
            push!(node.noise_terms, Noise(nothing, 1.0))
        end
        ctypes = JuMP.list_of_constraint_types(subproblem)
        node.has_integrality =
            (JuMP.VariableRef, MOI.Integer) in ctypes ||
            (JuMP.VariableRef, MOI.ZeroOne) in ctypes
    end
    for node_index in 1:graph.stages
        node = policy_graph.nodes[node_index]
        node.bellman_function =
            initialize_bellman_function(bellman_function, policy_graph, node)
    end
    for (k, v) in policy_graph.initial_root_state
        x_out = policy_graph[policy_graph.first_node].states[k].out
        if JuMP.has_lower_bound(x_out) && JuMP.lower_bound(x_out) > v
            error("Initial point $(v) violates lower bound on state $k")
        elseif JuMP.has_upper_bound(x_out) && JuMP.upper_bound(x_out) < v
            error("Initial point $(v) violates upper bound on state $k")
        end
    end
    domain = _get_incoming_domain(policy_graph)
    for (node_name, node) in policy_graph.nodes
        for (k, v) in domain[node_name]
            node.incoming_state_bounds[k] = something(v, (-Inf, Inf, false))
        end
    end
    return policy_graph
end

function _get_incoming_domain(model::PolicyGraph)
    function _bounds(x)
        l, u = -Inf, Inf
        if has_lower_bound(x)
            l = lower_bound(x)
        end
        if has_upper_bound(x)
            u = upper_bound(x)
        end
        if is_fixed(x)
            l = u = fix_value(x)
        end
        is_int = is_integer(x)
        if is_binary(x)
            l, u = max(something(l, 0.0), 0.0), min(something(u, 1.0), 1.0)
            is_int = true
        end
        return l, u, is_int
    end
    outgoing_bounds = Dict{Tuple{Int,Symbol},Any}(
        (k, state_name) => nothing for (k, node) in model.nodes for
        (state_name, _) in node.states
    )
    for (k, node) in model.nodes
        for noise in node.noise_terms
            parameterize(node, noise.term)
            for (state_name, state) in node.states
                domain = outgoing_bounds[(k, state_name)]
                l_new, u_new, is_int_new = _bounds(state.out)
                outgoing_bounds[(k, state_name)] = if domain === nothing
                    (l_new, u_new, is_int_new)
                else
                    l, u, is_int = domain::Tuple{Float64,Float64,Bool}
                    (min(l, l_new), max(u, u_new), is_int & is_int_new)
                end
            end
        end
    end
    incoming_bounds = Dict{Int,Dict{Symbol,Any}}()
    for (k, node) in model.nodes
        incoming_bounds[k] = Dict{Symbol,Any}(
            state_name => nothing for (state_name, _) in node.states
        )
    end
    for (parent_name, parent) in model.nodes
        for (state_name, state) in parent.states
            domain_new = outgoing_bounds[(parent_name, state_name)]
            l_new, u_new, is_int_new = domain_new::Tuple{Float64,Float64,Bool}
            parent.next_node === nothing && continue
            child = parent.next_node
            domain = incoming_bounds[child][state_name]
            incoming_bounds[child][state_name] = if domain === nothing
                (l_new, u_new, is_int_new)
            else
                l, u, is_int = domain::Tuple{Float64,Float64,Bool}
                (min(l, l_new), max(u, u_new), is_int & is_int_new)
            end
        end
    end
    for (state_name, value) in model.initial_root_state
        incoming_bounds[model.first_node][state_name] = (-Inf, Inf, false)
    end
    return incoming_bounds
end

function get_node(subproblem::JuMP.Model)
    return subproblem.ext[:MSUC_node]::Node
end

function get_policy_graph(subproblem::JuMP.Model)
    return subproblem.ext[:MSUC_policy_graph]::PolicyGraph
end

function parameterize(
    modify::Function,
    subproblem::JuMP.Model,
    realizations::AbstractVector{T},
    probability::AbstractVector{Float64} = fill(
        1.0 / length(realizations),
        length(realizations),
    ),
) where {T}
    node = get_node(subproblem)
    if length(node.noise_terms) != 0
        error("Duplicate calls to MSUC.parameterize detected.")
    end
    for (realization, prob) in zip(realizations, probability)
        push!(node.noise_terms, Noise(realization, prob))
    end
    node.parameterize = modify
    return
end

function set_stage_objective(
    subproblem::JuMP.Model,
    stage_objective::Union{Real,JuMP.AbstractJuMPScalar},
)
    node = get_node(subproblem)
    node.stage_objective = stage_objective
    node.stage_objective_set = false
    return
end

function set_stage_objective(::JuMP.Model, f)
    return error(
        "Unable to set the stage-objective of type $(typeof(f)). It must be " *
        "a scalar function.",
    )
end

macro stageobjective(subproblem, expr)
    code = MutableArithmetics.rewrite_and_return(expr)
    return quote
        MSUC.set_stage_objective($(esc(subproblem)), $code)
    end
end

function build_Φ(graph::PolicyGraph)
    Φ = Dict{Tuple{Int,Int},Float64}()
    Φ[(graph.root_node, graph.first_node)] = 1.0
    for (index, node) in graph.nodes
        node.next_node === nothing || (Φ[(index, node.next_node)] = 1.0)
    end
    return Φ
end
