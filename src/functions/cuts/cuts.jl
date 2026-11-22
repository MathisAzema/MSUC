#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# At a dispatch stage the schedule is fixed to x^k, so a cut reduces to
#     theta >= A + delta' zeta + b' y,
# and only the right-hand side A and the slopes b are rewritten when the schedule changes.

# Right-hand side `A` and per-unit slopes `b` of `cut` at `schedule`.
#   3-bin:    A = alpha + tau' x^k, b = beta.
#   interval: A = alpha + sum of tau_{a,b} over selected intervals, b_j = sum of beta_{a,b} over those of unit j.
function instantiate(cut::BinCut, schedule::Schedule, N::Int)
    return cut.alpha + LinearAlgebra.dot(cut.tau, schedule.x), cut.beta
end

function instantiate(cut::IntervalCut, schedule::Schedule, N::Int)
    A = cut.alpha
    b = zeros(N)
    for (id, j) in zip(schedule.selected, schedule.selected_units)
        A += cut.tau[id]
        b[j] += cut.beta[id]
    end
    return A, b
end

const CutRef = JuMP.ConstraintRef{
    JuMP.Model,
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}},
    JuMP.ScalarShape,
}

# Writes the right-hand sides and y-slopes of `cuts` at `schedule` in two batched solver calls, sending
# only slopes that differ from `cut.applied`. Per-coefficient updates were far slower than the LP solves.
function instantiate_cuts!(cuts::AbstractVector{<:AbstractCut}, y_out, schedule::Schedule, N::Int)
    isempty(cuts) && return
    rhs_refs, rhs = CutRef[], Float64[]
    slope_refs, slope_vars, slope_values = CutRef[], JuMP.VariableRef[], Float64[]
    for cut in cuts
        ref = cut.constraint_ref::CutRef
        A, b = instantiate(cut, schedule, N)
        push!(rhs_refs, ref)
        push!(rhs, A)
        first_time = isempty(cut.applied)
        for j in eachindex(y_out)
            if first_time || b[j] != cut.applied[j]
                push!(slope_refs, ref)
                push!(slope_vars, y_out[j])
                push!(slope_values, -b[j])
            end
        end
        if first_time
            cut.applied = copy(b)
        else
            copyto!(cut.applied, b)
        end
    end
    JuMP.set_normalized_rhs(rhs_refs, rhs)
    isempty(slope_refs) || JuMP.set_normalized_coefficient(slope_refs, slope_vars, slope_values)
    return
end

instantiate_cut!(cut::AbstractCut, y_out, schedule::Schedule, N::Int) =
    instantiate_cuts!([cut], y_out, schedule, N)

# At the root the schedule is a decision: the part of the cut not involving zeta or theta is written
# on the decision variables.
function root_term(cut::BinCut, node::Node, instance::Instance)
    expr = JuMP.AffExpr(0.0)
    x = commitment_variables(node.planning, instance)
    for k in eachindex(x)
        cut.tau[k] != 0.0 && JuMP.add_to_expression!(expr, cut.tau[k], x[k])
    end
    for j in eachindex(cut.beta)
        JuMP.add_to_expression!(expr, cut.beta[j], node.power[j].out)
    end
    return expr
end

# Interval formulation at the root: V(gamma, y~_0, zeta_0), with y~_{j,a,b,0} given by (16).
function root_term(cut::IntervalCut, node::Node, instance::Instance)
    expr = JuMP.AffExpr(0.0)
    idx = instance.intervals
    for id in eachindex(cut.tau)
        j, a = idx.unit[id], idx.a[id]
        coef = cut.tau[id] + cut.beta[id] * initial_interval_output(instance.Thermalunits[j], a)
        coef != 0.0 &&
            JuMP.add_to_expression!(expr, coef, node.planning.gamma[(j, a, idx.b[id])])
    end
    return expr
end

# Value of `cut` at the root for `schedule` (same expression as `add_cut!`/`root_term`), without the solver.
function root_cut_value(model::PolicyGraph, cut::BinCut, schedule::Schedule)
    start = initial_state(model.instance)
    return cut.alpha +
           LinearAlgebra.dot(cut.delta, start.zeta) +
           LinearAlgebra.dot(cut.tau, schedule.x) +
           LinearAlgebra.dot(cut.beta, start.power)
end

function root_cut_value(model::PolicyGraph, cut::IntervalCut, schedule::Schedule)
    start = initial_state(model.instance)
    idx = model.instance.intervals
    value = cut.alpha + LinearAlgebra.dot(cut.delta, start.zeta)
    for id in schedule.selected
        unit = model.instance.Thermalunits[idx.unit[id]]
        value += cut.tau[id] + cut.beta[id] * initial_interval_output(unit, idx.a[id])
    end
    return value
end

# Adds `cut` to `node`: written on the decision variables at the root, reduced to the current schedule at
# dispatch stages. If no schedule is set yet (reloading cuts into a fresh model), the next `set_schedule!`
# instantiates it.
function add_cut!(model::PolicyGraph, node::Node, cut::AbstractCut)
    subproblem = node.subproblem
    θ = bellman_term(node.bellman_function)
    ζ_term = sum(
        cut.delta[j] * node.zeta[j].out for j in eachindex(cut.delta);
        init = JuMP.AffExpr(0.0),
    )
    if node.index == 1
        cut.constraint_ref = @constraint(
            subproblem,
            θ >= cut.alpha + ζ_term + root_term(cut, node, model.instance)
        )
    else
        cut.constraint_ref = @constraint(subproblem, θ - ζ_term >= 0.0)
        model.schedule === nothing || instantiate_cut!(
            cut,
            [s.out for s in node.power],
            model.schedule,
            model.instance.N,
        )
    end
    push!(node.cuts, cut)
    return
end
