#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Position de u_{i,t}, v_{i,t}, w_{i,t} dans le vecteur plat x de `Schedule`.
commitment_dimension(instance::Instance) = instance.N * (3 * instance.TimeHorizon + 1)

u_index(instance::Instance, i, t) = i + instance.N * t

function v_index(instance::Instance, i, t)
    return instance.N * (instance.TimeHorizon + 1) + i + instance.N * (t - 1)
end

function w_index(instance::Instance, i, t)
    N, T = instance.N, instance.TimeHorizon
    return N * (T + 1) + N * T + i + N * (t - 1)
end

"""
    commitment_variables(root, instance)

Variables du planning de la racine, dans l'ordre du vecteur plat `Schedule.x`.
"""
function commitment_variables(root::RootVariables, instance::Instance)
    N, T = instance.N, instance.TimeHorizon
    return vcat(
        [root.u[i, t] for t in 0:T for i in 1:N],
        [root.v[i, t] for t in 1:T for i in 1:N],
        [root.w[i, t] for t in 1:T for i in 1:N],
    )
end

"""
    shutdown_cap(unit, schedule, i, t)

Enveloppe d'arret (46) de l'unite `i` a l'etape `t` : si l'unite s'arrete en b <= T, sa
production doit pouvoir redescendre jusqu'a MinPower, d'ou min{Pmax, Pmin + R^D (b-1-t)}.
Sans arret a venir, ou unite eteinte, la borne de capacite suffit : Pmax.
"""
function shutdown_cap(unit::ThermalUnit, schedule::Schedule, i::Int, t::Int)
    schedule.u[i, t+1] > 0.5 || return unit.MaxPower
    for b in (t+1):size(schedule.w, 2)
        if schedule.w[i, b] > 0.5
            return min(unit.MaxPower, unit.MinPower + unit.DeltaRampDown * (b - t - 1))
        end
    end
    return unit.MaxPower
end

"""
    set_stage_parameters!(params, instance, schedule, t)

Fixe les parametres du sous-probleme de l'etape `t` au planning `schedule`.
"""
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

"""
    extract_schedule(model)

Planning x^k lu dans la solution courante de la racine, arrondi a {0, 1} : les tolerances
entieres du solveur (1e-5) ne doivent pas se propager aux etapes de dispatch.
"""
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

"""
    set_schedule!(model, schedule)

Fixe le planning courant : parametres des etapes de dispatch, puis second membre et pentes
de chaque coupe stockee (qui dependent du planning).
"""
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

"""
    schedule_commitment_cost(instance, schedule)

Cout d'engagement (objectif du probleme racine) d'un planning : cout a vide, de demarrage et d'arret
sur tout l'horizon. `schedule.u` est indexe de t = 0 a T, `v` et `w` de t = 1 a T.
"""
function schedule_commitment_cost(instance::Instance, schedule::Schedule)
    return sum(
        unit.ConstTerm * schedule.u[unit.name, t+1] +
        unit.StartUpCost * schedule.v[unit.name, t] +
        unit.StartDownCost * schedule.w[unit.name, t] for unit in instance.Thermalunits for
        t in 1:instance.TimeHorizon
    )
end
