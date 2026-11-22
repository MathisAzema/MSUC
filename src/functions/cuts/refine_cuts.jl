#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    refine_bellman_function(model, node, items, y_k, ζ_k; interval_state = nothing)

Add to `node` the expected cut built from the next-stage subproblems solved at (y_k, ζ_k) for each noise
realization (`items`). `interval_state` (required for the interval formulation, ignored by 3-bin) is the
forward-pass interval state ytilde^k, the point where each interval's local problem is linearized.
"""
function refine_bellman_function(model::PolicyGraph, node::Node, items, y_k, ζ_k; interval_state = nothing)
    TimerOutputs.@timeit model.timer_output "compute_coefficients" begin
        if model.method == ExtendedFormulation
            interval_state === nothing &&
                error("la formulation intervalle a besoin de l'etat par intervalle de la passe avant")
            α, τ, β, δ = compute_coefficients_extended(model, node, items, y_k, ζ_k, interval_state)
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

# Coefficients (α, τ, β, δ) of the 3-bin cut (Prop. 2); τ is a vector over x = (u, v, w).
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

# Writes `Σ_k weights[k] * cut.field` into `total`. Zero-weight cuts (the non-binding ones, ~95%) are
# skipped. The `::Vector{Float64}` assertion is needed: `getfield` is not inferable and the result would
# be `Any`, making the caller's loop dynamically dispatched.
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

# Coefficients (α, τ, β, δ) of the interval cut (Prop. 4); τ and β are vectors over intervals. The outer
# multipliers (σ, ν) come from the projected 3-bin subproblem, the local coefficients from the closed form
# `local_support`, linearized at `interval_state` (ytilde^k from the forward pass).
function compute_coefficients_extended(model::PolicyGraph, node::Node, items, y_k, ζ_k, interval_state)
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

    δ = zeros(length(ζ_k))
    σ_mean = zeros(length(cuts))
    for i in 1:nscen
        δ .+= probability[i] .* items.duals[i].δ
        σ_mean .+= probability[i] .* items.duals[i].σ
    end

    # Propagation τ_prop = Σ_k σ̄_k τ_k; intervals inactive at stage t+1 keep τ_prop and β = 0.
    τ = _weighted_sum(cuts, σ_mean, :tau)
    β = zeros(nid)

    active = [id for id in 1:nid if is_on_interval(idx.a[id], idx.b[id], t + 1)]
    y_prev = [interval_state[id] for id in active]

    # Per noise ξ: z_ξ = c_j - ν_ξ + Σ_k σ_ξ,k β_k(a,b), then the affine support of the local problem at
    # `y_prev`, weighted by p_ξ: τ_ab = τ_prop + Σ_ξ p_ξ h_ξ, β_ab = Σ_ξ p_ξ β_ξ.
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
