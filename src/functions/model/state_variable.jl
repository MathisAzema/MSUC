#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors, Lea Kapelevich.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# JuMP variable extension for state variables, e.g.
#   @variable(subproblem, 0 <= x[i=1:2] <= i, MSUC.State, initial_value = i)
# creates `State(x[i]_in, x[i]_out)` objects; `JuMP.value.(x)` returns `State`s of values.

function JuMP.build_variable(
    _error::Function,
    info::JuMP.VariableInfo,
    ::Type{State};
    initial_value = NaN,
    kwargs...,
)
    if isnan(initial_value)
        _error(
            "When creating a state variable, you must set the " *
            "`initial_value` keyword to the value of the state variable at" *
            " the root node.",
        )
    end
    return StateInfo(
        JuMP.VariableInfo(
            false,
            NaN,  # lower bound
            false,
            NaN,  # upper bound
            false,
            NaN,  # fixed value
            false,
            NaN,  # start value
            false,
            false, # binary and integer
        ),
        info,
        initial_value,
        kwargs,
    )
end

function JuMP.add_variable(
    subproblem::JuMP.Model,
    state_info::StateInfo,
    name::String,
)
    state = State(
        JuMP.add_variable(
            subproblem,
            JuMP.ScalarVariable(state_info.in),
            name * "_in",
        ),
        JuMP.add_variable(
            subproblem,
            JuMP.ScalarVariable(state_info.out),
            name * "_out",
        ),
    )
    node = get_node(subproblem)
    sym_name = Symbol(name)
    @assert !haskey(node.states, sym_name)  # JuMP prevents duplicate names.
    node.states[sym_name] = state
    graph = get_policy_graph(subproblem)
    graph.initial_root_state[sym_name] = state_info.initial_value
    return state
end

function JuMP.value(state::State{JuMP.VariableRef})
    return State(JuMP.value(state.in), JuMP.value(state.out))
end

# Overload for broadcast syntax such as `JuMP.value.([state_1, state_2])`.
Broadcast.broadcastable(state::State{JuMP.VariableRef}) = Ref(state)

function JuMP.set_optimizer(model::MSUC.PolicyGraph, optimizer)
    for node in values(model.nodes)
        set_optimizer(node.subproblem, optimizer)
        set_silent(node.subproblem)
    end
    return
end
