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

# Ligne d'epigraphe d'une coupe de noeud d'etape : `theta - zeta_term >= A` plus `-b_j y_j`.
const CutRef = JuMP.ConstraintRef{
    JuMP.Model,
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}},
    JuMP.ScalarShape,
}

"""
    instantiate_cuts!(cuts, y_out, schedule, N)

Ecrit dans le sous-probleme le second membre et les pentes en y de `cuts` au planning `schedule`. Les
modifications sont envoyees au solveur en deux lots (seconds membres, puis pentes), et seules les
pentes qui different de celles deja ecrites (`cut.applied`) y sont envoyees : d'une iteration a la
suivante le planning ne change que pour quelques unites, et en 3-bin les pentes ne changent jamais. Envoyer
chaque coefficient separement coutait bien plus cher que la resolution des LP.
"""
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
decision a la racine et reduite au planning courant aux etapes de dispatch. Tant qu'aucun planning
n'est fixe (modele neuf dont on recharge des coupes), la ligne n'est pas instanciee : le prochain
`set_schedule!` s'en charge.
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
