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
    TimerOutputs.@timeit model.timer_output "compute_coefficients" begin
        if model.method == ExtendedFormulation
            α, τ, β, δ = compute_coefficients_extended(model, node, items, y_k, ζ_k)
            cut = IntervalCut(α, τ, β, δ, nothing)
        else
            α, τ, β, δ = compute_coefficients_3bin(model, node, items, y_k, ζ_k)
            cut = BinCut(α, τ, β, δ, nothing)
        end
    end
    TimerOutputs.@timeit model.timer_output "add_cut" begin
        add_cut!(model, node, cut)
    end
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
    interval_coefficients(unit, a, b, t, horizon, y_prev, τ_prop, z, probability) -> (τ_ab, β_ab)

Coefficients de la coupe intervalle pour l'intervalle [a:b] de `unit` (Prop. 4), actif a l'etape
t+1.
  * τ_ab = `τ_prop` (propagation des coupes du noeud suivant, ponderees par leurs multiplicateurs
    moyens)  +  Σ_ξ p_ξ h_ξ  (valeur locale).
  * β_ab = Σ_ξ p_ξ β_ξ  (pente locale en y).
Pour chaque alea ξ, `(h_ξ, β_ξ)` est le support affine du probleme local `local_support` en
`z[ξ]` = c_j - ν_ξ + Σ_k σ_ξ,k β_k(a,b), evalue au point de linearisation `y_prev` : la production
courante si l'intervalle est choisi, `linearization_point` (regle (55)) sinon. Pour un intervalle inactif a
l'etape t+1, le probleme local est nul : seule la propagation subsiste (`τ_prop`, β_ab = 0).
"""
function interval_coefficients(unit, a, b, t, horizon, y_prev, τ_prop, z, probability)
    τ_ab = τ_prop
    β_ab = 0.0
    for i in eachindex(probability)
        h, β_z = local_support(unit, a, b, t + 1, horizon, y_prev, z[i])
        τ_ab += probability[i] * h
        β_ab += probability[i] * β_z
    end
    return τ_ab, β_ab
end

# Somme ponderee `Σ_k weights[k] * cut.field` des coefficients des coupes (un vecteur sur les
# intervalles) : c'est la propagation des coupes du noeud suivant.
function _weighted_sum(cuts, weights, field::Symbol)
    total = zeros(length(getfield(cuts[1], field)))
    for (k, cut) in enumerate(cuts)
        total .+= weights[k] .* getfield(cut, field)
    end
    return total
end

# Coefficients (α, τ, β, δ) de la coupe intervalle (Prop. 4) : τ et β sont des vecteurs sur les
# intervalles (identifiants de `IntervalIndex`). Multiplicateurs exterieurs (σ, ν) du sous-probleme
# 3-bin projete, coefficients locaux par intervalle en forme close (`local_support`). Les
# sommes sur les coupes stockees sont faites par vecteurs entiers, pas intervalle par intervalle.
function compute_coefficients_extended(model::PolicyGraph, node::Node, items, y_k, ζ_k)
    instance = model.instance
    idx = instance.intervals
    units = instance.Thermalunits
    schedule = model.schedule
    cuts = model[node.index+1].cuts
    probability = items.probability
    nscen = length(probability)
    nid = length(idx.unit)
    t = node.index - 1
    horizon = instance.TimeHorizon

    # Pentes moyennes sur les aleas : δ (dual de l'etat entrant zeta, comme en 3-bin) et σ (duaux
    # des coupes du noeud suivant).
    δ = zeros(length(ζ_k))
    σ_mean = zeros(length(cuts))
    for i in 1:nscen
        δ .+= probability[i] .* items.duals[i].δ
        σ_mean .+= probability[i] .* items.duals[i].σ
    end

    # Propagation : τ_prop = Σ_k σ̄_k τ_k ; et z_ξ = c_j - ν_ξ + Σ_k σ_ξ,k β_k pour chaque alea.
    τ = _weighted_sum(cuts, σ_mean, :tau)
    z = zeros(nscen, nid)
    for i in 1:nscen
        d = items.duals[i]
        z[i, :] .= _weighted_sum(cuts, d.σ, :beta)
        for id in 1:nid
            unit = units[idx.unit[id]]
            z[i, id] = unit.LinearTerm - d.ν[unit.Bus] + z[i, id]
        end
    end

    # Partie locale, pour les seuls intervalles actifs a l'etape t+1.
    β = zeros(nid)
    is_selected = falses(nid)
    is_selected[schedule.selected] .= true
    for id in 1:nid
        a, b = idx.a[id], idx.b[id]
        is_on_interval(a, b, t + 1) || continue
        unit = units[idx.unit[id]]
        y_prev =
            is_selected[id] ? y_k[unit.name] :
            linearization_point(unit, a, b, t, horizon, y_k[unit.name])
        τ[id], β[id] = interval_coefficients(
            unit, a, b, t, horizon, y_prev, τ[id], view(z, :, id), probability,
        )
    end

    value = sum(probability[i] * items.objectives[i] for i in 1:nscen)
    α =
        value - LinearAlgebra.dot(δ, ζ_k) - sum(
            β[id] * y_k[idx.unit[id]] + τ[id] for id in schedule.selected;
            init = 0.0,
        )
    return α, τ, β, δ
end
