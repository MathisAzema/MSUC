#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Positions of u_{i,t}, v_{i,t}, w_{i,t} in the flat vector x of `Schedule`.
commitment_dimension(instance::Instance) = instance.N * (3 * instance.TimeHorizon + 1)

u_index(instance::Instance, i, t) = i + instance.N * t

function v_index(instance::Instance, i, t)
    return instance.N * (instance.TimeHorizon + 1) + i + instance.N * (t - 1)
end

function w_index(instance::Instance, i, t)
    N, T = instance.N, instance.TimeHorizon
    return N * (T + 1) + N * T + i + N * (t - 1)
end

function commitment_variables(root::RootVariables, instance::Instance)
    N, T = instance.N, instance.TimeHorizon
    return vcat(
        [root.u[i, t] for t in 0:T for i in 1:N],
        [root.v[i, t] for t in 1:T for i in 1:N],
        [root.w[i, t] for t in 1:T for i in 1:N],
    )
end

# Shutdown envelope (46) of unit `i` at stage `t`: if it shuts down at b <= T, its output must be
# able to ramp down to MinPower, i.e. min{Pmax, Pmin + R^D (b-1-t)}; otherwise Pmax.
function shutdown_cap(unit::ThermalUnit, schedule::Schedule, i::Int, t::Int)
    schedule.u[i, t+1] > 0.5 || return unit.MaxPower
    for b in (t+1):size(schedule.w, 2)
        if schedule.w[i, b] > 0.5
            return min(unit.MaxPower, unit.MinPower + unit.DeltaRampDown * (b - t - 1))
        end
    end
    return unit.MaxPower
end

function set_stage_parameters!(
    params::StageParameters,
    instance::Instance,
    schedule::Schedule,
    t::Int,
)
    for (i, unit) in enumerate(instance.Thermalunits)
        JuMP.fix(params.u_prev[i], schedule.u[i, t])
        JuMP.fix(params.u[i], schedule.u[i, t+1])
        JuMP.fix(params.v[i], schedule.v[i, t])
        JuMP.fix(params.w[i], schedule.w[i, t])
        for (h, p) in enumerate(params.w_future[i])
            JuMP.fix(p, schedule.w[i, t+h])
        end
        if !isempty(params.cap)
            JuMP.fix(params.cap[i], shutdown_cap(unit, schedule, i, t))
        end
    end
    return
end

# Schedule x^k read from the current root solution, rounded to {0, 1} so that the solver's
# integrality tolerance (1e-5) does not propagate to the dispatch stages.
function extract_schedule(model::PolicyGraph)
    instance = model.instance
    N, T = instance.N, instance.TimeHorizon
    root = model[model.first_node].planning::RootVariables
    u = [round(JuMP.value(root.u[i, t])) for i in 1:N, t in 0:T]
    v = [round(JuMP.value(root.v[i, t])) for i in 1:N, t in 1:T]
    w = [round(JuMP.value(root.w[i, t])) for i in 1:N, t in 1:T]
    keys = sort!([k for (k, var) in root.gamma if JuMP.value(var) > 0.5])
    selected = [instance.intervals.id[k] for k in keys]
    return Schedule(u, v, w, selected, [k[1] for k in keys])
end

# Fix the schedule in the dispatch stages, then update the right-hand side and slopes of every
# stored cut (they depend on the schedule).
function set_schedule!(model::PolicyGraph, schedule::Schedule)
    model.schedule = schedule
    N = model.instance.N
    for (node_index, node) in model.nodes
        node_index == 1 && continue
        set_stage_parameters!(node.planning, model.instance, schedule, node_index - 1)
        y_out = [s.out for s in node.power]
        instantiate_cuts!(node.cuts, y_out, schedule, N)
    end
    return
end

function schedule_commitment_cost(instance::Instance, schedule::Schedule)
    return sum(
        unit.ConstTerm * schedule.u[unit.name, t+1] +
        unit.StartUpCost * schedule.v[unit.name, t] +
        unit.StartDownCost * schedule.w[unit.name, t] for unit in instance.Thermalunits for
        t in 1:instance.TimeHorizon
    )
end
