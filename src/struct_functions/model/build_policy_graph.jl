#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct Graph
    stages::Int
end

struct Method
    string::String
end

ExtendedFormulation = Method("ExtendedFormulation")

SDDPFormulation = Method("SDDP")

mutable struct Node
    index::Int
    subproblem::JuMP.Model
    next_node::Union{Nothing,Int}
    noise_terms::Vector{Noise}
    parameterize::Function
    states::Dict{Symbol,State{JuMP.VariableRef}}
    power::Vector{State{JuMP.VariableRef}}
    zeta::Vector{State{JuMP.VariableRef}}
    stage_objective::Any
    stage_objective_set::Bool
    bellman_function::Any
    ext::Dict{Symbol,Any}
    incoming_state_bounds::Dict{Symbol,Tuple{Float64,Float64,Bool}}
    cuts::Vector{AbstractCut}
    demand_constraint::Vector{JuMP.ConstraintRef}
    planning::Any   # RootVariables at the root, StageParameters at stage nodes
    computation_time::Vector{Float64}
end

mutable struct PolicyGraph
    objective_sense::MOI.OptimizationSense
    root_node::Int
    first_node::Int
    initial_root_state::Dict{Symbol,Float64}
    nodes::Dict{Int,Node}
    most_recent_training_results::Union{Nothing,TrainingResults}
    ext::Dict{Symbol,Any}
    timer_output::TimerOutputs.TimerOutput

    instance::Instance
    method::Method
    force::Float64
    schedule::Union{Nothing,Schedule}   # current x^k, fixed during the iteration

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
            instance,
            method,
            force,
            nothing,
        )
    end
end
