#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# --- Generation des coupes --------------------------------------------------------------

"""
    refine_bellman_function(model, node, items, y_k, ζ_k)

Ajoute au noeud `node` la coupe moyenne construite a partir des sous-problemes de l'etape
suivante resolus en (y_k, ζ_k) pour chaque realisation de l'alea (`items`) : calcule les
coefficients (α, τ, β, δ) de la coupe selon la formulation du modele, puis l'ajoute (`add_cut!`).
"""
function refine_bellman_function(model::PolicyGraph, node::Node, items, y_k, ζ_k)
    if model.method == ExtendedFormulation
        α, τ, β, δ = compute_coefficients_extended(model, node, items, y_k, ζ_k)
        cut = IntervalCut(α, τ, β, δ, nothing)
    else
        α, τ, β, δ = compute_coefficients_3bin(model, node, items, y_k, ζ_k)
        cut = BinCut(α, τ, β, δ, nothing)
    end
    add_cut!(model, node, cut)
    return
end

# Coefficients (α, τ, β, δ) de la coupe 3-bin (Prop. 2) : τ est un vecteur sur x = (u, v, w).
# Les pentes en y et zeta sont les duaux des etats entrants, celles en x les duaux des parametres
# du planning, auxquels s'ajoute la propagation des coupes futures, dont le second membre depend
# de x par tau.
function compute_coefficients_3bin(model::PolicyGraph, node::Node, items, y_k, ζ_k)
    instance = model.instance
    schedule = model.schedule
    probability = items.probability
    next_cuts = model[node.index+1].cuts
    nscen = length(probability)

    τ = zeros(commitment_dimension(instance))
    β = zeros(instance.N)
    δ = zeros(length(ζ_k))
    σ_mean = zeros(length(next_cuts))
    for i in 1:nscen
        d = items.duals[i]
        τ .+= probability[i] .* d.τ
        β .+= probability[i] .* d.β
        δ .+= probability[i] .* d.δ
        σ_mean .+= probability[i] .* d.σ
    end
    for (k, cut) in enumerate(next_cuts)
        τ .+= σ_mean[k] .* cut.tau
    end

    value = sum(probability[i] * items.objectives[i] for i in 1:nscen)
    α =
        value - LinearAlgebra.dot(τ, schedule.x) - LinearAlgebra.dot(β, y_k) -
        LinearAlgebra.dot(δ, ζ_k)
    return α, τ, β, δ
end

"""
    interval_coefficients(model, node, items, σ_mean, y_k, unit, a, b) -> (τ_ab, β_ab)

Coefficients de la coupe intervalle pour l'intervalle [a:b] de `unit` (Prop. 4).
  * τ_ab = Σ_k σ̄_k τ_k(a,b)  (propagation des coupes du noeud suivant, ponderees par leurs
    multiplicateurs moyens)  +  Σ_ξ p_ξ h_ξ  (valeur locale).
  * β_ab = Σ_ξ p_ξ β_ξ  (pente locale en y).
Pour chaque alea ξ, `(h_ξ, β_ξ)` est le support affine du probleme local `local_support` en
z_ξ = c_j - ν_ξ + Σ_k σ_ξ,k β_k(a,b), evalue au point de linearisation `y_prev` : la production
courante si l'intervalle est choisi, sa trajectoire de reference sinon. Hors de l'intervalle
(t+1 non compris dans [a:b-1]) le probleme local est nul : seule la propagation subsiste.
"""
function interval_coefficients(model::PolicyGraph, node::Node, items, σ_mean, y_k, unit, a, b)
    t = node.index - 1
    horizon = model.instance.TimeHorizon
    cuts = model[node.index+1].cuts
    key = (unit.name, a, b)
    τ_ab = sum(σ_mean[k] * cut.tau[key] for (k, cut) in enumerate(cuts))
    β_ab = 0.0
    if is_on_interval(a, b, t + 1)
        y_prev =
            key in model.schedule.selected_set ? y_k[unit.name] :
            reference_output(unit, a, b, t, horizon)
        for (i, d) in enumerate(items.duals)
            z =
                unit.LinearTerm - d.ν[unit.Bus] +
                sum(d.σ[k] * cut.beta[key] for (k, cut) in enumerate(cuts))
            h, β_z = local_support(unit, a, b, t + 1, horizon, y_prev, z)
            τ_ab += items.probability[i] * h
            β_ab += items.probability[i] * β_z
        end
    end
    return τ_ab, β_ab
end

# Coefficients (α, τ, β, δ) de la coupe intervalle (Prop. 4) : τ et β sont des dictionnaires par
# intervalle (j, a, b). Multiplicateurs exterieurs (σ, ν) du sous-probleme 3-bin projete,
# coefficients locaux par intervalle en forme close (`local_support`).
function compute_coefficients_extended(model::PolicyGraph, node::Node, items, y_k, ζ_k)
    instance = model.instance
    schedule = model.schedule
    thermal_units = instance.Thermalunits
    cuts = model[node.index+1].cuts
    probability = items.probability
    nscen = length(probability)

    δ = zeros(length(ζ_k))
    σ_mean = zeros(length(cuts))
    for i in 1:nscen
        δ .+= probability[i] .* items.duals[i].δ
        σ_mean .+= probability[i] .* items.duals[i].σ
    end

    τ = Dict{Tuple{Int64,Int64,Int64},Float64}()
    β = Dict{Tuple{Int64,Int64,Int64},Float64}()
    for unit in thermal_units, (a, b) in unit.intervals
        τ[(unit.name, a, b)], β[(unit.name, a, b)] =
            interval_coefficients(model, node, items, σ_mean, y_k, unit, a, b)
    end

    value = sum(probability[i] * items.objectives[i] for i in 1:nscen)
    α =
        value - LinearAlgebra.dot(δ, ζ_k) - sum(
            β[(j, a, b)] * y_k[j] + τ[(j, a, b)] for
            (j, a, b) in schedule.selected;
            init = 0.0,
        )
    return α, τ, β, δ
end
