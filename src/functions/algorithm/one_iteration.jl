#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    one_iteration(model, options)

Une iteration de l'algorithme 2 : probleme racine, fixation du planning x^k pour les etapes de
dispatch, puis `options.passes_per_mip` passes avant/arriere (une seule par defaut) ; puis mise a jour
de la borne inferieure (maximum courant des bornes duales, donc valide et monotone), du journal et
des criteres d'arret.

Par defaut (`train(...; reuse_incumbent = true)`), le MIP racine n'est pas resolu si le planning precedent, evalue
avec la nouvelle approximation, est deja dans l'ecart d'optimalite de la meilleure borne
(`reusable_incumbent`) : le planning est conserve tel quel, les coupes ajoutees depuis ayant deja ete
instanciees a ce planning par `add_cut!`, et la borne reste celle de l'iteration precedente.

Avec plusieurs passes, le planning x^k reste fixe : les passes ne resolvent que des LP et affinent les
coupes de chaque etape avant de resoudre de nouveau le MIP racine, qui est le poste le plus couteux.
Les coupes sont valides pour tout planning ; pour ne pas gonfler le MIP, seule la coupe de la derniere
passe est ajoutee a la racine.
"""
function one_iteration(model::PolicyGraph, options::Options)
    model.ext[:numerical_issue] = false
    reused = nothing
    if get(model.ext, :reuse_incumbent, true)
        TimerOutputs.@timeit model.timer_output "first_stage_value" begin
            reused = reusable_incumbent(model)
        end
    end
    first_stage_solved = reused === nothing
    if first_stage_solved
        TimerOutputs.@timeit model.timer_output "solve_first_stage_problem" begin
            lower_bound, incumbent = solve_first_stage_problem(model, options.initial_state)
        end
        # Le planning x^k lu a la racine est fixe pour toute l'iteration : parametres des etapes de
        # dispatch, puis second membre et pentes des coupes stockees (qui dependent de x^k).
        TimerOutputs.@timeit model.timer_output "set_schedule" begin
            set_schedule!(model, extract_schedule(model))
        end
    else
        lower_bound, incumbent = reused.bound, reused.incumbent
        push!(model[model.first_node].computation_time, 0.0)    # une entree par iteration : pas de MIP resolu
    end
    for pass in 1:options.passes_per_mip
        TimerOutputs.@timeit model.timer_output "forward_pass" begin
            forward_trajectory = forward_pass(model, options)
            options.forward_pass_callback(forward_trajectory)
        end
        TimerOutputs.@timeit model.timer_output "backward_pass" begin
            backward_pass(
                model,
                forward_trajectory.sampled_states_power,
                forward_trajectory.sampled_states_zeta;
                root_cut = pass == options.passes_per_mip,
                sampled_states_interval = forward_trajectory.sampled_states_interval,
            )
        end
    end
    bound = max(get(model.ext, :best_bound, -Inf), lower_bound)
    model.ext[:best_bound] = bound
    push!(
        options.log,
        Log(
            length(options.log) + 1,
            bound,
            incumbent,
            time() - options.start_time,
            model.ext[:total_solves],
            model.ext[:numerical_issue],
            first_stage_solved,
        ),
    )
    # Criteres d'arret : nombre d'iterations et temps total.
    status = :not_solved
    if options.iteration_limit !== nothing &&
       options.log[end].iteration >= options.iteration_limit
        status = :iteration_limit
    elseif options.time_limit !== nothing &&
           options.log[end].time >= options.time_limit
        status = :time_limit
    end
    return IterationResult(bound, status, model.ext[:numerical_issue])
end
