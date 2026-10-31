#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    backward_pass(model, sampled_states_power, sampled_states_zeta; root_cut = true)

Passe arriere (algorithme 2, lignes 12-20) : en remontant de la fin vers la racine, resout l'etape
suivante au point `(y, zeta)` visite par la passe avant, pour chaque realisation de l'alea, et en
tire une coupe pour le noeud courant. Le graphe est lineaire : le noeud d'indice k est a la
profondeur k de la trajectoire. Avec `root_cut = false`, la coupe de la racine (une ligne dense du MIP)
n'est pas ajoutee.
"""
function backward_pass(
    model::PolicyGraph,
    sampled_states_power::Vector{Vector{Float64}},
    sampled_states_zeta::Vector{Vector{Float64}};
    root_cut::Bool = true,
)
    for index in length(sampled_states_power):-1:1
        node = model[index]
        node.next_node === nothing && continue
        index == model.first_node && !root_cut && continue
        state = (power = sampled_states_power[index], zeta = sampled_states_zeta[index])
        items = BackwardPassItems()
        solve_next_node(model, node, items, state)
        refine_bellman_function(model, node, items, state.power, state.zeta)
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
