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

Avec `train(...; reuse_incumbent = true)` (faux par defaut), le MIP racine n'est pas resolu si le planning precedent, evalue
avec la nouvelle approximation, est deja dans l'ecart d'optimalite de la meilleure borne
(`reusable_incumbent`) : le planning est conserve tel quel, les coupes ajoutees depuis ayant deja ete
instanciees a ce planning par `add_cut!`, et la borne reste celle de l'iteration precedente.

Avec plusieurs passes, le planning x^k reste fixe : les passes ne resolvent que des LP et affinent les
coupes de chaque etape avant de resoudre de nouveau le MIP racine, qui est le poste le plus couteux.
Les coupes sont valides pour tout planning ; pour ne pas gonfler le MIP, seule la coupe de la derniere
passe est ajoutee a la racine.

Avec `train(...; time_limit)`, le MIP racine est limite au temps restant, `time_limit` moins le temps ecoule
(sauf a la premiere iteration) : s'il l'atteint, sa borne duale reste valide et le planning x^k est le meilleur
trouve, ou le planning precedent si aucun n'a ete trouve. Si le temps est alors ecoule, l'iteration s'arrete sans
passe (`model.ext[:passes_skipped]`), sauf la premiere : l'entrainement ne deborde que de la duree du MIP au-dela de la
limite. Sinon, les passes ont lieu ensuite.
"""
function one_iteration(model::PolicyGraph, options::Options)
    model.ext[:numerical_issue] = false
    reused = nothing
    if get(model.ext, :reuse_incumbent, false)
        TimerOutputs.@timeit model.timer_output "first_stage_value" begin
            reused = reusable_incumbent(model)
        end
    end
    first_stage_solved = reused === nothing
    if first_stage_solved
        # Avec une limite de temps, le MIP racine n'a que le temps restant (T_max - t_now) : une resolution
        # longue ne fait pas deborder l'entrainement. La premiere iteration n'est pas plafonnee : sans planning
        # precedent, il faut en obtenir un.
        remaining =
            options.time_limit === nothing || model.schedule === nothing ? nothing :
            options.time_limit - (time() - options.start_time)
        TimerOutputs.@timeit model.timer_output "solve_first_stage_problem" begin
            lower_bound, incumbent =
                solve_first_stage_problem(model, options.initial_state; time_limit = remaining)
        end
        if incumbent === nothing
            # Temps ecoule sans planning realisable : on garde le planning precedent, evalue avec la nouvelle
            # approximation ; la borne duale du MIP reste valide.
            incumbent = first_stage_value(model, model.schedule)
        else
            # Le planning x^k lu a la racine est fixe pour toute l'iteration : parametres des etapes de
            # dispatch, puis second membre et pentes des coupes stockees (qui dependent de x^k).
            TimerOutputs.@timeit model.timer_output "set_schedule" begin
                set_schedule!(model, extract_schedule(model))
            end
        end
    else
        lower_bound, incumbent = reused.bound, reused.incumbent
        push!(model[model.first_node].computation_time, 0.0)    # une entree par iteration : pas de MIP resolu
    end
    # Limite de temps atteinte apres le MIP (typiquement : plafonne par le temps restant) : l'iteration s'arrete la,
    # sans passe. Sa borne et son planning x^k sont acquis, et ses coupes ne serviraient a aucune politique evaluee a
    # temps : celle de l'iteration k utilise les coupes jusqu'a l'iteration k-1. La premiere iteration va toujours au bout.
    out_of_time =
        options.time_limit !== nothing &&
        !isempty(options.log) &&
        time() - options.start_time >= options.time_limit
    model.ext[:passes_skipped] = out_of_time
    for pass in 1:(out_of_time ? 0 : options.passes_per_mip)
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
