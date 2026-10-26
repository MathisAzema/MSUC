#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

mutable struct IterationLimit <: AbstractStoppingRule
    limit::Int
end

stopping_rule_status(::IterationLimit) = :iteration_limit

function convergence_test(::PolicyGraph, log::Vector{Log}, rule::IterationLimit)
    return log[end].iteration >= rule.limit
end

mutable struct TimeLimit <: AbstractStoppingRule
    limit::Float64
end

stopping_rule_status(::TimeLimit) = :time_limit

function convergence_test(::PolicyGraph, log::Vector{Log}, rule::TimeLimit)
    return log[end].time >= rule.limit
end

struct BoundStalling <: AbstractStoppingRule
    num_previous_iterations::Int
    tolerance::Float64
end

stopping_rule_status(::BoundStalling) = :bound_stalling

function convergence_test(
    ::PolicyGraph,
    log::Vector{Log},
    rule::BoundStalling,
)
    if length(log) < rule.num_previous_iterations + 1
        return false
    end
    existing_solves = log[1].total_solves > log[end].total_solves / length(log)
    if !existing_solves && isapprox(log[1].bound, log[end].bound; atol = 1e-6)
        return all(l -> isapprox(l.bound, l.simulation_value; atol = 1e-6), log)
    end
    for i in 1:rule.num_previous_iterations
        if abs(log[end-i].bound - log[end-i+1].bound) > rule.tolerance
            return false
        end
    end
    return true
end

mutable struct SimulationStoppingRule{F} <: AbstractStoppingRule
    simulator::F
    replications::Int
    period::Int
    data::Vector{Any}
    last_iteration::Int
    distance_tol::Float64
    bound_tol::Float64
end

function SimulationStoppingRule(;
    sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
    replications::Int = -1,
    period::Int = -1,
    distance_tol::Float64 = 1e-2,
    bound_tol::Float64 = 1e-4,
)
    cached_sampling_scheme =
        PSRSamplingScheme(replications; sampling_scheme = sampling_scheme)
    function simulator(model, N)
        cached_sampling_scheme.N = max(N, cached_sampling_scheme.N)
        scenarios = simulate(
            model,
            N;
            sampling_scheme = cached_sampling_scheme,
            parallel_scheme = Serial(),
        )
        keys = [:stage_objective, :bellman_term]
        return map(scenarios) do scenario
            return [getindex.(scenario, k) for k in keys]
        end
    end
    return SimulationStoppingRule(
        simulator,
        replications,
        period,
        Any[],
        0,
        distance_tol,
        bound_tol,
    )
end

stopping_rule_status(::SimulationStoppingRule) = :simulation_stopping

function _compute_distance(x::Real, y::Real)
    if x ≈ y
        return 0.0
    end
    return abs(x - y) / max(1.0, abs(x), abs(y))
end

function _compute_distance(new_data::Vector, old_data::Vector)
    d = sum(_compute_distance(x, y)^2 for (x, y) in zip(new_data, old_data))
    return sqrt(d)
end

function _period(period, iterations)
    if period != -1
        return period
    elseif iterations <= 100
        return 20
    elseif iterations <= 1_000
        return 100
    else
        return 500
    end
end

function convergence_test(
    model::PolicyGraph,
    log::Vector{Log},
    rule::SimulationStoppingRule,
)
    if rule.replications == -1
        rule.replications = min(100, _unique_paths(model))
    end
    if isempty(rule.data)
        rule.data = rule.simulator(model, rule.replications)
        rule.last_iteration = 0
        return false
    end
    if length(log) <= 5
        return false
    end
    if !isapprox(
        log[end].bound,
        log[end-5].bound;
        atol = rule.bound_tol,
        rtol = rule.bound_tol,
    )
        return false
    end
    if length(log) - rule.last_iteration < _period(rule.period, length(log))
        return false
    end
    new_data = rule.simulator(model, rule.replications)
    distance = _compute_distance(new_data, rule.data)
    rule.data = new_data
    rule.last_iteration = length(log)
    return distance < rule.distance_tol
end
