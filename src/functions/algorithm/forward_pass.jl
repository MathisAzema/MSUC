#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    forward_pass(model, options)

Passe avant (algorithme 2, lignes 6-11) : tire une trajectoire d'aleas et resout chaque etape de
dispatch, en partant de l'etat initial. Suppose la racine resolue et le planning x^k fixe
(`one_iteration`). Renvoie les etats sortants le long de la trajectoire, racine comprise, qui
sont (y, zeta) a la sortie du noeud k.

En formulation intervalle, la passe avant lit aussi les multiplicateurs de chaque probleme d'etape et
construit l'etat par intervalle ytilde^k (`forward_interval_state`, regle des prix de la Sect. 4.2) :
`sampled_states_interval[k]` est le vecteur des ytilde^k_{a,b} de tous les intervalles a la sortie du
noeud k, dans l'ordre de `instance.intervals`. La passe arriere linearise les coupes en ces points. Pour les
diagnostics, `sampled_signals_interval[k]` donne les signaux de cout z^F de la meme etape (NaN pour un intervalle
inactif) ; ces deux champs valent NaN a la racine.
En formulation 3-bin, il n'y a pas d'etat par intervalle : `nothing`.
"""
function forward_pass(model::PolicyGraph, options::Options)
    TimerOutputs.@timeit model.timer_output "sample_scenario" begin
        noises = sample_scenario(model)
    end
    state = options.initial_state
    sampled_states_power = [state.power]
    sampled_states_zeta = [state.zeta]
    price_rule = model.method == ExtendedFormulation
    interval_state = price_rule ? initial_interval_state(model.instance) : nothing
    sampled_states_interval = price_rule ? [interval_state] : nothing
    sampled_signals_interval = price_rule ? [fill(NaN, length(interval_state))] : nothing

    for node_index in 2:length(noises)    # le noeud 1 est la racine
        TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
            subproblem_results = solve_subproblem(
                model,
                model[node_index],
                state,
                noises[node_index];
                compute_duals = price_rule,
            )
        end
        state = subproblem_results.state
        push!(sampled_states_power, state.power)
        push!(sampled_states_zeta, state.zeta)
        if price_rule
            TimerOutputs.@timeit model.timer_output "interval_state" begin
                interval_state, signal = forward_interval_state(
                    model,
                    model[node_index],
                    interval_state,
                    state.power,
                    subproblem_results.duals,
                )
            end
            push!(sampled_states_interval, interval_state)
            push!(sampled_signals_interval, signal)
        end
    end
    return (
        sampled_states_power = sampled_states_power,
        sampled_states_zeta = sampled_states_zeta,
        sampled_states_interval = sampled_states_interval,
        sampled_signals_interval = sampled_signals_interval,
    )
end
