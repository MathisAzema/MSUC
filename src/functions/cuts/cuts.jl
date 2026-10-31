#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# --- Coupes au planning courant -------------------------------------------------------
#
# A l'etape de dispatch le planning est fixe a x^k : la coupe se reduit a
#     theta >= A + delta' zeta + b' y,   A = alpha + (terme en x^k),   b_j = coefficient de y_j,
# dont on ne reecrit que le second membre A et les coefficients b a chaque nouveau planning.

"""
    instantiate(cut, schedule, N) -> (A, b)

Second membre `A` et pentes `b` (une par unite) de la coupe au planning `schedule`.
  * 3-bin      : A = alpha + tau' x^k, b = beta.
  * intervalle : A = alpha + somme des tau_{a,b} des intervalles choisis,
                 b_j = somme des beta_{a,b} de ceux de l'unite j.
"""
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

function instantiate_cut!(cut::AbstractCut, y_out, schedule::Schedule, N::Int)
    A, b = instantiate(cut, schedule, N)
    JuMP.set_normalized_rhs(cut.constraint_ref, A)
    JuMP.set_normalized_coefficient.(cut.constraint_ref, y_out, -b)
    return
end

# A la racine le planning est variable : la partie de la coupe qui ne depend ni de zeta ni
# de theta s'ecrit sur les variables de decision.
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

# Formulation intervalle a la racine : V(gamma, y~_0, zeta_0), ou y~_{j,a,b,0} vaut (17).
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

"""
    add_cut!(model, node, cut)

Ajoute `cut` aux coupes du noeud : cree sa ligne d'epigraphe, ecrite sur les variables de
decision a la racine et reduite au planning courant aux etapes de dispatch.
"""
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
        instantiate_cut!(
            cut,
            [s.out for s in node.power],
            model.schedule,
            model.instance.N,
        )
    end
    push!(node.cuts, cut)
    return
end
