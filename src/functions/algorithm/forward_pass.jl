#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    forward_pass(model, options)

Passe avant (algorithme 2, lignes 6-11) : tire une trajectoire d'aleas et resout chaque etape de
dispatch, en partant de l'etat initial. Suppose la racine resolue et le planning x^k fixe
(`one_iteration`). Renvoie les etats sortants le long de la trajectoire, racine comprise, qui
sont les points d'evaluation de la passe arriere : `sampled_states_power[k]` et `sampled_states_zeta[k]`
sont (y, zeta) a la sortie du noeud k.
"""
function forward_pass(model::PolicyGraph, options::Options)
    TimerOutputs.@timeit model.timer_output "sample_scenario" begin
        noises = sample_scenario(model)
    end
    state = options.initial_state
    sampled_states_power = [state.power]
    sampled_states_zeta = [state.zeta]

    for node_index in 2:length(noises)    # le noeud 1 est la racine
        TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
            subproblem_results = solve_subproblem(
                model,
                model[node_index],
                state,
                noises[node_index],
            )
        end
        state = subproblem_results.state
        push!(sampled_states_power, state.power)
        push!(sampled_states_zeta, state.zeta)
    end
    return (
        sampled_states_power = sampled_states_power,
        sampled_states_zeta = sampled_states_zeta,
    )
end
