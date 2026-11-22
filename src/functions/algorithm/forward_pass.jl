#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    forward_pass(model, options)

Forward pass (Algorithm 2, lines 6-12): sample a noise trajectory and solve each dispatch stage from the
initial state. Assumes the root is solved and the schedule x^k is fixed (see `one_iteration`). Returns
the outgoing states `(y, zeta)` along the trajectory, root included.

With the interval formulation it also builds the interval states ytilde^k (`forward_interval_state`, rule
`interval_rule` of `train`): `sampled_states_interval[k]` holds ytilde^k for all intervals at the exit of
node k, in the order of `instance.intervals`, and `sampled_signals_interval[k]` the cost signals z^F (for
diagnostics). Both are `nothing` with the 3-bin formulation.
"""
function forward_pass(model::PolicyGraph, options::Options)
    TimerOutputs.@timeit model.timer_output "sample_scenario" begin
        noises = sample_scenario(model)
    end
    state = options.initial_state
    sampled_states_power = [state.power]
    sampled_states_zeta = [state.zeta]
    interval = model.method == ExtendedFormulation
    rule = get(model.ext, :interval_rule, :dual)
    interval_state = interval ? initial_interval_state(model.instance) : nothing
    sampled_states_interval = interval ? [interval_state] : nothing
    sampled_signals_interval = interval ? [fill(NaN, length(interval_state))] : nothing

    for node_index in 2:length(noises)
        TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
            subproblem_results = solve_subproblem(
                model,
                model[node_index],
                state,
                noises[node_index];
                compute_duals = interval && rule == :dual,
            )
        end
        state = subproblem_results.state
        push!(sampled_states_power, state.power)
        push!(sampled_states_zeta, state.zeta)
        if interval
            TimerOutputs.@timeit model.timer_output "interval_state" begin
                interval_state, signal = forward_interval_state(
                    model,
                    model[node_index],
                    interval_state,
                    state.power,
                    subproblem_results.duals,
                    rule,
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
