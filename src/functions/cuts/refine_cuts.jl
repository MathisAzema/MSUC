#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# --- Generation des coupes --------------------------------------------------------------

"""
    refine_bellman_function(model, node, items, y_k, ζ_k; interval_state = nothing)

Ajoute au noeud `node` la coupe moyenne construite a partir des sous-problemes de l'etape
suivante resolus en (y_k, ζ_k) pour chaque realisation de l'alea (`items`) : calcule les
coefficients (α, τ, β, δ) de la coupe selon la formulation du modele, puis l'ajoute (`add_cut!`).
`interval_state` (formulation intervalle) donne l'etat par intervalle ytilde^k a la sortie du noeud, construit
par la passe avant (regle des prix) ; sans lui, c'est `linearization_point` qui choisit les points.
"""
function refine_bellman_function(model::PolicyGraph, node::Node, items, y_k, ζ_k; interval_state = nothing)
    TimerOutputs.@timeit model.timer_output "compute_coefficients" begin
        if model.method == ExtendedFormulation
            α, τ, β, δ = compute_coefficients_extended(model, node, items, y_k, ζ_k; interval_state)
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

# Somme ponderee `Σ_k weights[k] * cut.field` des coefficients des coupes (un vecteur sur les
# intervalles), ecrite dans `total` : c'est la propagation des coupes du noeud suivant. Les coupes de
# poids nul sont ignorees (le resultat est identique) : les poids sont les multiplicateurs des coupes du
# noeud suivant, nuls pour toutes celles qui ne sont pas saturees, soit environ 95 % d'entre elles.
# `getfield(cut, field)` n'est pas inferable : son type est affirme, sans quoi le resultat est `Any` et
# toute la boucle de `compute_coefficients_extended` qui l'utilise passe par du dispatch dynamique.
function _weighted_sum!(total::Vector{Float64}, cuts, weights, field::Symbol)
    fill!(total, 0.0)
    for (k, cut) in enumerate(cuts)
        iszero(weights[k]) && continue
        total .+= weights[k] .* getfield(cut, field)::Vector{Float64}
    end
    return total
end

function _weighted_sum(cuts, weights, field::Symbol)
    n = length(getfield(cuts[1], field)::Vector{Float64})
    return _weighted_sum!(zeros(n), cuts, weights, field)
end

# Coefficients (α, τ, β, δ) de la coupe intervalle (Prop. 4) : τ et β sont des vecteurs sur les
# intervalles (identifiants de `IntervalIndex`). Multiplicateurs exterieurs (σ, ν) du sous-probleme
# 3-bin projete, coefficients locaux par intervalle en forme close (`local_support`). Les
# sommes sur les coupes stockees sont faites par vecteurs entiers, pas intervalle par intervalle.
# `interval_state`, s'il est donne, remplace `linearization_point` : c'est l'etat par intervalle ytilde^k
# a la sortie du noeud, construit par la passe avant (regle des prix, `forward_interval_state`).
function compute_coefficients_extended(
    model::PolicyGraph,
    node::Node,
    items,
    y_k,
    ζ_k;
    interval_state = nothing,
)
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

    # Propagation : τ_prop = Σ_k σ̄_k τ_k. Un intervalle inactif a l'etape t+1 n'a pas de probleme local :
    # il garde τ_prop et β = 0.
    τ = _weighted_sum(cuts, σ_mean, :tau)
    β = zeros(nid)

    # Partie locale, pour les seuls intervalles actifs a l'etape t+1 : point de linearisation ỹ de chacun.
    # Regle par defaut (`linearization_point`) : la production courante si l'intervalle est choisi ou
    # atteignable depuis un intervalle choisi de la meme unite, la production maximale atteignable ȳ sinon.
    # Regle des prix : l'etat par intervalle de la passe avant, `interval_state`.
    active = [id for id in 1:nid if is_on_interval(idx.a[id], idx.b[id], t + 1)]
    y_prev = zeros(length(active))
    if interval_state === nothing
        is_selected = falses(nid)
        is_selected[schedule.selected] .= true
        chosen = [Tuple{Int,Int}[] for _ in 1:instance.N]    # intervalles choisis (c, d) de chaque unite
        for id in schedule.selected
            push!(chosen[idx.unit[id]], (idx.a[id], idx.b[id]))
        end
        for (n, id) in enumerate(active)
            unit = units[idx.unit[id]]
            y_prev[n] = linearization_point(
                unit, idx.a[id], idx.b[id], t, horizon, y_k[unit.name], is_selected[id], chosen[unit.name],
            )
        end
    else
        for (n, id) in enumerate(active)
            y_prev[n] = interval_state[id]
        end
    end

    # Pour chaque alea ξ : z_ξ = c_j - ν_ξ + Σ_k σ_ξ,k β_k(a,b), puis le support affine du probleme local
    # en `y_prev` (`local_support`), pondere par p_ξ : τ_ab = τ_prop + Σ_ξ p_ξ h_ξ et β_ab = Σ_ξ p_ξ β_ξ.
    # Un seul vecteur `σβ` sert pour tous les aleas, sans matrice intermediaire.
    σβ = zeros(nid)
    for i in 1:nscen
        d = items.duals[i]
        _weighted_sum!(σβ, cuts, d.σ, :beta)
        for (n, id) in enumerate(active)
            unit = units[idx.unit[id]]
            z = unit.LinearTerm - d.ν[unit.Bus] + σβ[id]
            h, β_z = local_support(unit, idx.a[id], idx.b[id], t + 1, horizon, y_prev[n], z)
            τ[id] += probability[i] * h
            β[id] += probability[i] * β_z
        end
    end

    value = sum(probability[i] * items.objectives[i] for i in 1:nscen)
    α =
        value - LinearAlgebra.dot(δ, ζ_k) - sum(
            β[id] * y_k[idx.unit[id]] + τ[id] for id in schedule.selected;
            init = 0.0,
        )
    return α, τ, β, δ
end
