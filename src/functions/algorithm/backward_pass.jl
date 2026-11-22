#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    backward_pass(model, sampled_states_power, sampled_states_zeta; root_cut = true,
                  sampled_states_interval = nothing)

Backward pass (Algorithm 2, lines 13-20): from the last stage to the root, solve the next stage at the
state visited by the forward pass for every noise realization and build a cut for the current node.
With `root_cut = false` the (dense) root cut is not added. With `sampled_states_interval`, interval
cut coefficients are linearized at the forward-pass interval states.
"""
function backward_pass(
    model::PolicyGraph,
    sampled_states_power::Vector{Vector{Float64}},
    sampled_states_zeta::Vector{Vector{Float64}};
    root_cut::Bool = true,
    sampled_states_interval = nothing,
)
    for index in length(sampled_states_power):-1:1
        node = model[index]
        node.next_node === nothing && continue
        index == model.first_node && !root_cut && continue
        state = (power = sampled_states_power[index], zeta = sampled_states_zeta[index])
        interval_state = sampled_states_interval === nothing ? nothing : sampled_states_interval[index]
        items = BackwardPassItems()
        solve_next_node(model, node, items, state)
        refine_bellman_function(model, node, items, state.power, state.zeta; interval_state)
    end
    return
end

function solve_next_node(model::PolicyGraph, node::Node, items::BackwardPassItems, state)
    next_node = model[node.next_node::Int]
    for noise in next_node.noise_terms
        TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
            subproblem_results = solve_subproblem(
                model,
                next_node,
                state,
                noise.term;
                compute_duals = true,
            )
        end
        push!(items.duals, subproblem_results.duals)
        push!(items.probability, noise.probability)
        push!(items.objectives, subproblem_results.objective)
    end
    return
end
