#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct DefaultForwardPass <: AbstractForwardPass
    include_last_node::Bool
    function DefaultForwardPass(; include_last_node::Bool = true)
        return new(include_last_node)
    end
end

function forward_pass(
    model::PolicyGraph,
    options::Options,
    pass::DefaultForwardPass,
)
    instance = model.instance
    @_timeit_threadsafe model.timer_output "sample_scenario" begin
        scenario_path, _ = sample_scenario(model, options.sampling_scheme)
    end
    sampled_states = Dict{Symbol,Float64}[]
    sampled_states_power = Vector{Float64}[]
    sampled_states_zeta = Vector{Float64}[]
    sampled_states_gamma = Dict{Tuple{Int,Int,Int},Float64}[]
    selected_intervals = Tuple{Int,Int,Int}[]
    lower_bound = 0.0

    cumulative_value = 0.0

    incoming_state_value = copy(options.initial_state)

    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = model[node_index]
        lock(node.lock)
        try
            @_timeit_threadsafe model.timer_output "solve_subproblem" begin
                subproblem_results = solve_subproblem(
                    model,
                    node,
                    incoming_state_value,
                    noise,
                    scenario_path[1:depth];
                    duality_handler = nothing,
                )
            end
            cumulative_value += subproblem_results.stage_objective
            incoming_state_value = copy(subproblem_results.state)
            incoming_state_gamma = Dict{Tuple{Int,Int,Int},Float64}()
            for (k, var) in node.states_gamma
                incoming_state_gamma[k] = value(var)
                if value(var)>0.5 && node.index == 1
                    push!(selected_intervals, k)
                end
            end
            if node.index == 1
                reinstantiate_interval_cuts!(model, selected_intervals)
            end

            incoming_state_power = Float64[]
            for unit in model.instance.Thermalunits
                var_name = "power[$(unit.name)]"
                var_symbol = Symbol(var_name)
                push!(incoming_state_power, JuMP.value(JuMP.value(node.states_power[var_symbol].out)))
            end

            push!(sampled_states, incoming_state_value)
            push!(sampled_states_power, incoming_state_power)
            push!(sampled_states_zeta, [JuMP.value(z.out) for z in node.states_zeta])
            push!(sampled_states_gamma, incoming_state_gamma)

            if node_index == 1
                lower_bound = JuMP.objective_bound(node.subproblem)
            end
        finally
            unlock(node.lock)
        end
    end
    return (
        scenario_path = scenario_path,
        sampled_states = sampled_states,
        sampled_states_power = sampled_states_power,
        sampled_states_zeta = sampled_states_zeta,
        sampled_states_gamma = sampled_states_gamma,
        selected_intervals = selected_intervals,
        cumulative_value = cumulative_value,
        lower_bound = lower_bound,
    )
end

function reinstantiate_interval_cuts!(
    model::PolicyGraph,
    selected_intervals::Vector{Tuple{Int,Int,Int}},
)
    N = model.instance.N
    for (node_name, node) in model.nodes
        node_name == 1 && continue
        for cut in node.interval_cuts
            cstr = cut.constraint_ref
            set_normalized_rhs(
                cstr,
                cut.alpha +
                sum(cut.tau[j, a, b] for (j, a, b) in selected_intervals; init = 0.0),
            )
            set_normalized_coefficient.(
                cstr,
                [node.states[Symbol("power[$(i)]")].out for i in 1:N],
                [
                    -sum(
                        cut.beta[j, a, b] for (j, a, b) in selected_intervals if j == i;
                        init = 0.0
                    ) for i in 1:N
                ],
            )
        end
    end
    return
end

mutable struct RiskAdjustedForwardPass{F,T} <: AbstractForwardPass
    forward_pass::F
    risk_measure::T
    resampling_probability::Float64
    rejection_count::Int
    objectives::Vector{Float64}
    nominal_probability::Vector{Float64}
    adjusted_probability::Vector{Float64}
    archive::Vector{Any}
    resample_count::Vector{Int}
end

function RiskAdjustedForwardPass(;
    forward_pass::AbstractForwardPass,
    risk_measure::AbstractRiskMeasure,
    resampling_probability::Float64,
    rejection_count::Int = 5,
)
    if !(0 < resampling_probability < 1)
        throw(ArgumentError("Resampling probability must be in `(0, 1)`"))
    end
    return RiskAdjustedForwardPass{typeof(forward_pass),typeof(risk_measure)}(
        forward_pass,
        risk_measure,
        resampling_probability,
        rejection_count,
        Float64[],
        Float64[],
        Float64[],
        Any[],
        Int[],
    )
end

function forward_pass(
    model::PolicyGraph,
    options::Options,
    fp::RiskAdjustedForwardPass,
)
    if length(fp.archive) > 0 && rand() < fp.resampling_probability
        r = rand()
        for i in 1:length(fp.adjusted_probability)
            r -= fp.adjusted_probability[i]
            if r > 1e-8
                continue
            end
            pass = fp.archive[i]
            if fp.resample_count[i] >= fp.rejection_count
                splice!(fp.objectives, i)
                splice!(fp.nominal_probability, i)
                splice!(fp.adjusted_probability, i)
                splice!(fp.archive, i)
                splice!(fp.resample_count, i)
            else
                fp.resample_count[i] += 1
            end
            return pass
        end
    end
    pass = forward_pass(model, options, fp.forward_pass)
    push!(fp.objectives, pass.cumulative_value)
    push!(fp.nominal_probability, 0.0)
    fill!(fp.nominal_probability, 1 / length(fp.nominal_probability))
    push!(fp.adjusted_probability, 0.0)
    push!(fp.archive, pass)
    push!(fp.resample_count, 1)
    adjust_probability(
        fp.risk_measure,
        fp.adjusted_probability,
        fp.nominal_probability,
        fp.objectives,
        fp.objectives,
        model.objective_sense == MOI.MIN_SENSE,
    )
    return pass
end
