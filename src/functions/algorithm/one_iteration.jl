#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    one_iteration(model, options)

One iteration of Algorithm 2: solve the root problem, fix the schedule x^k for the dispatch stages, run
`options.passes_per_mip` forward/backward passes, then update the lower bound (running maximum of the
dual bounds, hence valid and monotone), the log and the stopping criteria.

With `reuse_incumbent = true` (see `train`), the root MIP is skipped when the previous schedule, evaluated
with the new approximation, is within the MIP gap of the best bound (`reusable_incumbent`).

With several passes the schedule stays fixed and only LPs are solved; only the cut of the last pass is
added to the root, to keep the MIP small.

With `time_limit`, the root MIP gets the remaining time (except at the first iteration); its dual bound
stays valid, and x^k is the best schedule found, or the previous one if none. If time is then up, the
iteration stops without passes (`model.ext[:passes_skipped]`), except the first one.
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
        # The first iteration is uncapped: there is no previous schedule to fall back on.
        remaining =
            options.time_limit === nothing || model.schedule === nothing ? nothing :
            options.time_limit - (time() - options.start_time)
        TimerOutputs.@timeit model.timer_output "solve_first_stage_problem" begin
            lower_bound, incumbent =
                solve_first_stage_problem(model, options.initial_state; time_limit = remaining)
        end
        if incumbent === nothing
            # Time limit hit without a feasible schedule: keep the previous one.
            incumbent = first_stage_value(model, model.schedule)
        else
            TimerOutputs.@timeit model.timer_output "set_schedule" begin
                set_schedule!(model, extract_schedule(model))
            end
        end
    else
        lower_bound, incumbent = reused.bound, reused.incumbent
        push!(model[model.first_node].computation_time, 0.0)
    end
    # Out of time after the MIP: skip the passes. The policy of iteration k uses cuts up to iteration k-1,
    # so these cuts would never be evaluated. The first iteration always runs its passes.
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
