#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function backward_pass(
    model::PolicyGraph,
    options::Options,
    scenario_path::Vector{Tuple{Int,NoiseType}},
    sampled_states::Vector{Dict{Symbol,Float64}},
    selected_intervals::Vector{Tuple{Int, Int, Int}},
    sampled_states_power::Vector{Vector{Float64}},
    sampled_states_zeta::Vector{Vector{Float64}},
    sampled_states_gamma::Vector{Dict{Tuple{Int, Int,Int},Float64}},
) where {NoiseType}
    cuts = Dict{Int,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    for index in length(scenario_path):-1:1
        outgoing_state = sampled_states[index]
        outgoing_state_power = sampled_states_power[index]
        outgoing_state_zeta = sampled_states_zeta[index]
        outgoing_state_gamma = sampled_states_gamma[index]
        node_index, _ = scenario_path[index]
        node = model[node_index]
        if node.next_node === nothing
            continue
        end
        if model.method == SDDPFormulation
            items = BackwardPassItems(Noise)
            solve_next_node(
                model,
                node,
                items,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[1:index],
                options.duality_handler,
                options,
            )
            new_cuts = refine_bellman_function(
                model,
                node,
                node.bellman_function,
                outgoing_state,
                items.duals,
                items.supports,
                items.probability,
                items.objectives,
            )
            push!(cuts[node_index], new_cuts)
        elseif model.method == ExtendedFormulation
            items = BackwardPassItems_extended(Noise)
            solve_next_node_extended(
                model,
                node,
                items,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[1:index],
                options.duality_handler,
                options,
            )

            refine_bellman_function_extended(
                model,
                node,
                selected_intervals,
                outgoing_state_power,
                outgoing_state_gamma,
                outgoing_state_zeta,
                items.σ,
                items.ν,
                items.probability,
                items.objectives,
            )

        else
            error("Unknown method: $(model.method).")
        end
    end
    return cuts
end

struct BackwardPassItems{U}
    cached_solutions::Dict{Tuple{Int,Any},Int}
    duals::Vector{Dict{Symbol,Float64}}
    supports::Vector{U}
    nodes::Vector{Int}
    probability::Vector{Float64}
    objectives::Vector{Float64}
    function BackwardPassItems(U)
        return new{U}(
            Dict{Tuple{Int,Any},Int}(),
            Dict{Symbol,Float64}[],
            U[],
            Int[],
            Float64[],
            Float64[],
        )
    end
end

struct BackwardPassItems_extended{U}
    cached_solutions::Dict{Tuple{Int,Any},Int}
    σ::Vector{Vector{Float64}}
    ν::Vector{Vector{Float64}}
    supports::Vector{U}
    nodes::Vector{Int}
    probability::Vector{Float64}
    objectives::Vector{Float64}
    function BackwardPassItems_extended(U)
        return new{U}(
            Dict{Tuple{Int,Any},Int}(),
            Vector{Float64}[],
            Vector{Float64}[],
            U[],
            Int[],
            Float64[],
            Float64[],
        )
    end
end

function solve_next_node(
    model::PolicyGraph,
    node::Node,
    items::BackwardPassItems,
    outgoing_state::Dict{Symbol,Float64},
    backward_sampling_scheme::AbstractBackwardSamplingScheme,
    scenario_path,
    duality_handler::Union{Nothing,AbstractDualityHandler},
    options,
)
    length_scenario_path = length(scenario_path)
    next_index = node.next_node::Int
    next_node = model[next_index]
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality = prepare_backward_pass(
            next_node,
            options.duality_handler,
            options,
        )
    end
    for noise in sample_backward_noise_terms_with_state(
        backward_sampling_scheme,
        next_node,
        outgoing_state,
    )
        if length(scenario_path) == length_scenario_path
            push!(scenario_path, (next_index, noise.term))
        else
            scenario_path[end] = (next_index, noise.term)
        end
        if haskey(items.cached_solutions, (next_index, noise.term))
            sol_index = items.cached_solutions[(next_index, noise.term)]
            push!(items.duals, items.duals[sol_index])
            push!(items.supports, items.supports[sol_index])
            push!(items.nodes, next_node.index)
            push!(items.probability, items.probability[sol_index])
            push!(items.objectives, items.objectives[sol_index])
        else
            TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
                subproblem_results = solve_subproblem(
                    model,
                    next_node,
                    outgoing_state,
                    noise.term,
                    scenario_path;
                    duality_handler = duality_handler,
                )
            end
            push!(items.duals, subproblem_results.duals)
            push!(items.supports, noise)
            push!(items.nodes, next_node.index)
            push!(
                items.probability,
                noise.probability,
            )
            push!(items.objectives, subproblem_results.objective)
            items.cached_solutions[(next_index, noise.term)] =
                length(items.duals)
        end
    end
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality()
    end
    if length(scenario_path) != length_scenario_path
        pop!(scenario_path)
    end
    return
end

function solve_next_node_extended(
    model::PolicyGraph,
    node::Node,
    items::BackwardPassItems_extended,
    outgoing_state::Dict{Symbol,Float64},
    backward_sampling_scheme::AbstractBackwardSamplingScheme,
    scenario_path,
    duality_handler::Union{Nothing,AbstractDualityHandler},
    options,
)
    length_scenario_path = length(scenario_path)
    next_index = node.next_node::Int
    next_node = model[next_index]
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality = prepare_backward_pass(
            next_node,
            options.duality_handler,
            options,
        )
    end
    for noise in sample_backward_noise_terms_with_state(
        backward_sampling_scheme,
        next_node,
        outgoing_state,
    )
        if length(scenario_path) == length_scenario_path
            push!(scenario_path, (next_index, noise.term))
        else
            scenario_path[end] = (next_index, noise.term)
        end
        if haskey(items.cached_solutions, (next_index, noise.term))
            sol_index = items.cached_solutions[(next_index, noise.term)]
            push!(items.σ, items.σ[sol_index])
            push!(items.ν, items.ν[sol_index])
            push!(items.supports, items.supports[sol_index])
            push!(items.nodes, next_node.index)
            push!(items.probability, items.probability[sol_index])
            push!(items.objectives, items.objectives[sol_index])
        else
            TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
                subproblem_results = solve_subproblem_extended(
                    model,
                    next_node,
                    outgoing_state,
                    noise.term,
                    scenario_path;
                    duality_handler = duality_handler,
                )
            end

            push!(items.supports, noise)
            push!(items.nodes, next_node.index)
            push!(items.σ, subproblem_results.σ)
            push!(items.ν, subproblem_results.ν)
            push!(
                items.probability,
                noise.probability,
            )
            push!(items.objectives, subproblem_results.objective)
            items.cached_solutions[(next_index, noise.term)] =
                length(items.σ)
        end
    end
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality()
    end
    if length(scenario_path) != length_scenario_path
        pop!(scenario_path)
    end
    return
end
