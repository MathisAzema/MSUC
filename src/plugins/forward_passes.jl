#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    DefaultForwardPass(; include_last_node::Bool = true)

The default forward pass.

If `include_last_node = false` and the sample terminated due to a cycle, then
the last node (which forms the cycle) is omitted. This can be useful option to
set when training, but it comes at the cost of not knowing which node formed the
cycle (if there are multiple possibilities).
"""
struct DefaultForwardPass <: AbstractForwardPass
    include_last_node::Bool
    function DefaultForwardPass(; include_last_node::Bool = true)
        return new(include_last_node)
    end
end

function forward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultForwardPass,
) where {T}
    # Le graphe est un chemin (LinearGraph) : l'echantillonnage ne peut pas boucler.
    instance = model.instance
    @_timeit_threadsafe model.timer_output "sample_scenario" begin
        scenario_path, _ = sample_scenario(model, options.sampling_scheme)
    end
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict{Symbol,Float64}[]
    sampled_states_power = Vector{Float64}[]
    sampled_states_zeta = Vector{Float64}[]
    sampled_states_gamma = Dict{Tuple{Int,Int,Int},Float64}[]
    selected_intervals = Tuple{Int,Int,Int}[]
    lower_bound = 0.0

    # A cumulator for the stage-objectives.
    cumulative_value = 0.0

    # Our initial incoming state.
    incoming_state_value = copy(options.initial_state)

    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = model[node_index]
        lock(node.lock)
        try
            # Solve the subproblem, note that `duality_handler = nothing`.
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
            # Cumulate the stage_objective.
            cumulative_value += subproblem_results.stage_objective
            # Set the outgoing state value as the incoming state value for the next
            # node.
            incoming_state_value = copy(subproblem_results.state)
            # Add the outgoing state variable to the list of states we have sampled
            # on this forward pass.
            incoming_state_gamma = Dict{Tuple{Int,Int,Int},Float64}()
            for (k, var) in node.states_gamma
                incoming_state_gamma[k] = value(var)
                if value(var)>0.5 && node.index == 1
                    # println("Gamma > 0.5 detected at node ", node.index, " for state ", k)
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

            # println((node.index, value(node.bellman_function.global_theta.theta), stage_objective_value(node, node.stage_objective), objective_value(node.subproblem)))

            if node_index == 1
                # Borne inferieure de l'iteration : probleme racine (26) avec
                # l'approximation courante (Alg. 2, ligne 5). C'est un MIP, donc on prend
                # sa borne DUALE : l'incumbent majore l'optimum de la tolerance MIPGap et
                # ne serait pas une borne valide. La trajectoire, elle, utilise l'incumbent.
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

"""
    reinstantiate_interval_cuts!(model, selected_intervals)

Reecrit chaque coupe intervalle (39) deja stockee, au planning `selected_intervals`.

C'est le mecanisme central de l'implementation : les coupes gardent leurs coefficients
PAR INTERVALLE (`tau`, `beta`), et la ligne d'epigraphe effective du sous-probleme est
reconstruite en sommant sur les intervalles selectionnes. On obtient ainsi (41g) sans
jamais porter les produits gamma * ytilde comme variables.

A appeler des que le planning change : apres la resolution de la racine dans la passe
avant, et avant toute simulation de politique.
"""
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

"""
    RiskAdjustedForwardPass(;
        forward_pass::AbstractForwardPass,
        risk_measure::AbstractRiskMeasure,
        resampling_probability::Float64,
        rejection_count::Int = 5,
    )

A forward pass that resamples a previous forward pass with
`resampling_probability` probability, and otherwise samples a new forward pass
using `forward_pass`.

The forward pass to revisit is chosen based on the risk-adjusted (using
`risk_measure`) probability of the cumulative stage objectives.

Note that this objective corresponds to the _first_ time we visited the
trajectory. Subsequent visits may have improved things, but we don't have the
mechanisms in-place to update it. Therefore, remove the forward pass from
resampling consideration after `rejection_count` revisits.
"""
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
                # We've explored this pass too many times. Kick it out of the
                # archive.
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

"""
    ImportanceSamplingForwardPass()

A forward pass that explores according to the risk adjusted sampling scheme
proposed in:

Dias Garcia, J., Leal, I., Chabar, R., and Pereira, M.V. (2023). A Multicut
Approach to Compute Upper Bounds for Risk-Averse MSUC.
https://arxiv.org/abs/2307.13190
"""
struct ImportanceSamplingForwardPass <: AbstractForwardPass end

function forward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::ImportanceSamplingForwardPass,
) where {T}
    scenario_path = Tuple{T,Any}[]
    sampled_states = Dict{Symbol,Float64}[]
    cumulative_value = 0.0
    incoming_state_value = copy(options.initial_state)
    node_index = sample_noise(model.root_children)
    node = model[node_index]
    noise = sample_noise(node.noise_terms)
    while node_index !== nothing
        node = model[node_index]
        lock(node.lock)
        try
            push!(scenario_path, (node_index, noise))
            subproblem_results = solve_subproblem(
                model,
                node,
                incoming_state_value,
                noise,
                scenario_path;
                duality_handler = nothing,
            )
            cumulative_value += subproblem_results.stage_objective
            incoming_state_value = copy(subproblem_results.state)
            push!(sampled_states, incoming_state_value)
            if isempty(node.bellman_function.local_thetas)
                # First iteration with no multi-cuts, or a node with no children
                node_index = MSUC.sample_noise(node.children)
                if node_index !== nothing
                    new_node = model[node_index]
                    noise = MSUC.sample_noise(new_node.noise_terms)
                end
            else
                objectives = map(node.bellman_function.local_thetas) do t
                    return JuMP.value(t.theta)
                end
                adjusted_probability = fill(NaN, length(objectives))
                nominal_probability = Float64[]
                support = Any[]
                for child in node.children
                    for noise in model[child.term].noise_terms
                        push!(
                            nominal_probability,
                            child.probability * noise.probability,
                        )
                        push!(support, (child.term, noise.term))
                    end
                end
                @assert length(nominal_probability) == length(objectives)
                _ = MSUC.adjust_probability(
                    options.risk_measures[node_index],
                    adjusted_probability,
                    nominal_probability,
                    support,
                    objectives,
                    model.objective_sense == MOI.MIN_SENSE,
                )
                terms = MSUC.Noise.(support, adjusted_probability)
                node_index, noise = MSUC.sample_noise(terms)
            end
        finally
            unlock(node.lock)
        end
    end
    return (
        scenario_path = scenario_path,
        sampled_states = sampled_states,
        cumulative_value = cumulative_value,
    )
end
