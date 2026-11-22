#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# 3-bin slopes on x for stage `t`: duals of the schedule parameters, placed in the flat vector `Schedule.x`.
function commitment_duals(params::StageParameters, instance::Instance, t::Int)
    τ = zeros(commitment_dimension(instance))
    for i in 1:instance.N
        τ[u_index(instance, i, t-1)] += JuMP.dual(JuMP.FixRef(params.u_prev[i]))
        τ[u_index(instance, i, t)] += JuMP.dual(JuMP.FixRef(params.u[i]))
        τ[v_index(instance, i, t)] += JuMP.dual(JuMP.FixRef(params.v[i]))
        τ[w_index(instance, i, t)] += JuMP.dual(JuMP.FixRef(params.w[i]))
        for (h, p) in enumerate(params.w_future[i])
            τ[w_index(instance, i, t+h)] += JuMP.dual(JuMP.FixRef(p))
        end
    end
    return τ
end

# Set to `false` to read cut duals one by one with `JuMP.dual` (generic, slower fallback).
const BULK_CUT_DUALS = Ref(true)

# Set to `true` to check every bulk read against `JuMP.dual` and error on any mismatch.
const CHECK_BULK_CUT_DUALS = Ref(false)

# Inner Gurobi optimizer of the subproblem, or `nothing` for another solver.
function _gurobi_optimizer(subproblem::JuMP.Model)
    inner = JuMP.unsafe_backend(subproblem)
    return inner isa Gurobi.Optimizer ? inner : nothing
end

# 0-based row of the cut in the Gurobi model.
_cut_row(inner::Gurobi.Optimizer, cut) =
    Cint(Gurobi._info(inner, JuMP.optimizer_index(cut.constraint_ref)).row - 1)

# Gurobi rows of the cuts of `node`, cached in `node.ext` since looking up a row costs far more than
# reading the duals. If the solver copies the whole model (numerical recovery, or a modified JuMP
# `Parameter`), rows get reordered: when the last cached row is stale, the whole cache is rebuilt.
function _cut_rows(node::Node, inner::Gurobi.Optimizer)
    rows = get!(() -> Cint[], node.ext, :cut_rows)::Vector{Cint}
    if !isempty(rows) && _cut_row(inner, node.cuts[length(rows)]) != rows[end]
        empty!(rows)
    end
    for k in (length(rows)+1):length(node.cuts)
        push!(rows, _cut_row(inner, node.cuts[k]))
    end
    return rows
end

# Duals (σ) of the cuts of `node` in the order of `node.cuts`. With Gurobi they are read in one call (the
# `Pi` vector) instead of one `JuMP.dual` per cut, which dominated the backward pass. Same values and sign
# as `JuMP.dual` (Gurobi.jl returns `Pi` for minimization, `-Pi` for maximization). Falls back to
# per-cut reads for other solvers, infeasibility certificates or `BULK_CUT_DUALS[] = false`.
function _cut_duals(node::Node)
    inner = BULK_CUT_DUALS[] ? _gurobi_optimizer(node.subproblem) : nothing
    if inner === nothing || inner.has_infeasibility_cert
        return [JuMP.dual(cut.constraint_ref) for cut in node.cuts]
    end
    rows = _cut_rows(node, inner)
    numconstrs = Ref{Cint}()
    Gurobi._check_ret(inner, Gurobi.GRBgetintattr(inner, "NumConstrs", numconstrs))
    pi = get!(() -> Cdouble[], node.ext, :pi_buffer)::Vector{Cdouble}
    length(pi) < numconstrs[] && resize!(pi, numconstrs[])
    Gurobi._check_ret(inner, Gurobi.GRBgetdblattrarray(inner, "Pi", 0, numconstrs[], pi))
    sign = MOI.get(inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE ? 1.0 : -1.0
    duals = [sign * pi[row+1] for row in rows]
    if CHECK_BULK_CUT_DUALS[]
        reference = [JuMP.dual(cut.constraint_ref) for cut in node.cuts]
        duals == reference ||
            error("cut duals of node $(node.index): bulk read differs from JuMP.dual")
    end
    return duals
end

# Multipliers of the stage subproblem (`StageDuals`). With the interval formulation, `β` and `τ` are
# computed in closed form (`local_support`) and left empty here.
function get_dual_solution(node::Node)
    if !_has_dual_solution(node)
        model = node.subproblem.ext[:MSUC_policy_graph]
        attempt_numerical_recovery(model, node; require_dual = true)
    end
    model = get_policy_graph(node.subproblem)
    extended = model.method == ExtendedFormulation
    β = extended ? Float64[] : [JuMP.dual(JuMP.FixRef(s.in)) for s in node.power]
    τ =
        extended ? Float64[] :
        commitment_duals(node.planning, model.instance, node.index - 1)
    return StageDuals(
        _cut_duals(node),
        JuMP.dual.(node.demand_constraint),
        β,
        [JuMP.dual(JuMP.FixRef(s.in)) for s in node.zeta],
        τ,
    )
end
