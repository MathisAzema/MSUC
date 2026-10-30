#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    one_iteration(model, options)

Une iteration de l'algorithme 2 : probleme racine, fixation du planning x^k pour les etapes de
dispatch, passe avant, passe arriere ; puis mise a jour de la borne inferieure (maximum courant
des bornes duales, donc valide et monotone), du journal et des criteres d'arret.
"""
function one_iteration(model::PolicyGraph, options::Options)
    model.ext[:numerical_issue] = false
    TimerOutputs.@timeit model.timer_output "solve_first_stage_problem" begin
        lower_bound = solve_first_stage_problem(model, options.initial_state)
    end
    # Le planning x^k lu a la racine est fixe pour toute l'iteration : parametres des etapes de
    # dispatch, puis second membre et pentes des coupes stockees (qui dependent de x^k).
    TimerOutputs.@timeit model.timer_output "set_schedule" begin
        set_schedule!(model, extract_schedule(model))
    end
    TimerOutputs.@timeit model.timer_output "forward_pass" begin
        forward_trajectory = forward_pass(model, options)
        options.forward_pass_callback(forward_trajectory)
    end
    TimerOutputs.@timeit model.timer_output "backward_pass" begin
        backward_pass(
            model,
            forward_trajectory.sampled_states_power,
            forward_trajectory.sampled_states_zeta,
        )
    end
    bound = max(get(model.ext, :best_bound, -Inf), lower_bound)
    model.ext[:best_bound] = bound
    push!(
        options.log,
        Log(
            length(options.log) + 1,
            bound,
            time() - options.start_time,
            model.ext[:total_solves],
            model.ext[:numerical_issue],
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
