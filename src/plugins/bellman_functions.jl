#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

mutable struct Cut
    intercept::Float64
    coefficients::Dict{Symbol,Float64}
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
end

mutable struct ConvexApproximation
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    cuts::Vector{Cut}

    function ConvexApproximation(
        theta::JuMP.VariableRef,
        states::Dict{Symbol,JuMP.VariableRef},
    )
        return new(theta, states, Cut[])
    end
end

_magnitude(x) = abs(x) > 0 ? log10(abs(x)) : 0

function _dynamic_range_warning(intercept, coefficients)
    lo = hi = _magnitude(intercept)
    lo_v = hi_v = intercept
    for v in values(coefficients)
        i = _magnitude(v)
        if v < lo_v
            lo, lo_v = i, v
        elseif v > hi_v
            hi, hi_v = i, v
        end
    end
    if hi - lo > 10
        @warn(
            """Found a cut with a mix of small and large coefficients.
          The order of magnitude difference is $(hi - lo).
          The smallest cofficient is $(lo_v).
          The largest coefficient is $(hi_v).

      You can ignore this warning, but it may be an indication of numerical issues.

      Consider rescaling your model by using different units, e.g, kilometers instead
      of meters. You should also consider reducing the accuracy of your input data (if
      you haven't already). For example, it probably doesn't make sense to measure the
      inflow into a reservoir to 10 decimal places.""",
            maxlog = 1,
        )
    end
    return
end

function _add_cut(
    V::ConvexApproximation,
    θᵏ::Float64,
    πᵏ::Dict{Symbol,Float64},
    xᵏ::Dict{Symbol,Float64},
)
    for (key, x) in xᵏ
        θᵏ -= πᵏ[key] * x
    end
    _dynamic_range_warning(θᵏ, πᵏ)
    cut = Cut(θᵏ, πᵏ, nothing)
    _add_cut_constraint_to_model(V, cut)
    push!(V.cuts, cut)
    return
end

function _add_cut_constraint_to_model(V::ConvexApproximation, cut::Cut)
    model = JuMP.owner_model(V.theta)
    expr = @expression(
        model,
        V.theta - sum(cut.coefficients[i] * x for (i, x) in V.states)
    )
    cut.constraint_ref = if JuMP.objective_sense(model) == MOI.MIN_SENSE
        @constraint(model, expr >= cut.intercept)
    else
        @constraint(model, expr <= cut.intercept)
    end
    return
end

@enum(CutType, SINGLE_CUT, MULTI_CUT)

# Internal struct: this struct is just a cache for arguments until we can build
# an actual instance of the type T at a later point.
struct InstanceFactory{T}
    args::Any
    kwargs::Any
    InstanceFactory{T}(args...; kwargs...) where {T} = new{T}(args, kwargs)
end

"""
    BellmanFunction

A representation of the value function. MSUC.jl uses the following unique
representation of the value function that is undocumented in the literature.

It supports three types of state variables:

 1) x - convex "resource" states
 2) y - concave "objective" states

In addition, we have three types of cuts:

 1) Single-cuts (also called "average" cuts in the literature), which involve
    the risk-adjusted expectation of the cost-to-go.
 2) Multi-cuts, which use a different cost-to-go term for each realization w.
 3) Risk-cuts, which correspond to the facets of the dual interpretation of a
    convex risk measure.

Therefore, ValueFunction returns a JuMP model of the following form:

```
V(x, b, y) =
    min: μᵀb + νᵀy + θ
    s.t. # "Single" / "Average" cuts
         μᵀb(j) + νᵀy(j) + θ >= α(j) + xᵀβ(j),          ∀ j ∈ J
         # "Multi" cuts
         μᵀb(k) + νᵀy(k) + φ(w) >= α(k, w) + xᵀβ(k, w), ∀w ∈ Ω, k ∈ K
         # "Risk-set" cuts
         θ ≥ Σ{p(k, w) * φ(w)}_w - μᵀb(k) - νᵀy(k),     ∀ k ∈ K
```
"""
mutable struct BellmanFunction
    cut_type::CutType
    global_theta::ConvexApproximation
    local_thetas::Vector{ConvexApproximation}
    # Cuts defining the dual representation of the risk measure.
    risk_set_cuts::Set{Vector{Float64}}
end

"""
    BellmanFunction(;
        lower_bound = -Inf,
        upper_bound = Inf,
        cut_type::CutType = MULTI_CUT,
    )
"""
function BellmanFunction(;
    lower_bound = -Inf,
    upper_bound = Inf,
    cut_type::CutType = MULTI_CUT,
)
    return InstanceFactory{BellmanFunction}(;
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        cut_type = cut_type,
    )
end

function bellman_term(bellman_function::BellmanFunction)
    return bellman_function.global_theta.theta
end

function initialize_bellman_function(
    factory::InstanceFactory{BellmanFunction},
    model::PolicyGraph{T},
    node::Node{T},
) where {T}
    lower_bound, upper_bound, cut_type = -Inf, Inf, SINGLE_CUT
    if length(factory.args) > 0
        error(
            "Positional arguments $(factory.args) ignored in BellmanFunction.",
        )
    end
    for (kw, value) in factory.kwargs
        if kw == :lower_bound
            lower_bound = value
        elseif kw == :upper_bound
            upper_bound = value
        elseif kw == :cut_type
            cut_type = value
        else
            error(
                "Keyword $(kw) not recognised as argument to BellmanFunction.",
            )
        end
    end
    if lower_bound == -Inf && upper_bound == Inf
        error("You must specify a finite bound on the cost-to-go term.")
    end
    if length(node.children) == 0
        lower_bound = upper_bound = 0.0
    end
    Θᴳ = @variable(node.subproblem)
    # lower_bound > -Inf && JuMP.set_lower_bound(Θᴳ, lower_bound)
    # upper_bound < Inf && JuMP.set_upper_bound(Θᴳ, upper_bound)

    # Coupe initiale nulle : V^{ext,0} = 0, de la forme (27).
    cstr = @constraint(node.subproblem, Θᴳ >= 0.0)
    thermal_units = model.instance.Thermalunits
    τ = Dict{Tuple{Int64,Int64,Int64},Float64}()
    β = Dict{Tuple{Int64,Int64,Int64},Float64}()
    for unit in thermal_units, (a, b) in unit.intervals
        τ[(unit.name, a, b)] = 0.0
        β[(unit.name, a, b)] = 0.0
    end
    push!(node.interval_cuts, IntervalCut(0.0, τ, β, zeros(length(node.states_zeta)), cstr))

    x′ = Dict(key => var.out for (key, var) in node.states)
    return BellmanFunction(
        cut_type,
        ConvexApproximation(Θᴳ, x′),
        ConvexApproximation[],
        Set{Vector{Float64}}(),
    )
end

function refine_bellman_function(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    risk_measure::AbstractRiskMeasure,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
) where {T}
    lock(node.lock)
    try
        return _refine_bellman_function_no_lock(
            model,
            node,
            bellman_function,
            risk_measure,
            outgoing_state,
            dual_variables,
            noise_supports,
            nominal_probability,
            objective_realizations,
        )
    finally
        unlock(node.lock)
    end
end

# Internal function: construit la coupe intervalle (39)-(40) au point d'essai et
# l'ajoute au sous-probleme du noeud (Alg. 2, lignes 17-34).
#
# Pour chaque unite j et chaque intervalle [a:b] de T_j :
#
#   z_{j,a,b}   = c_{j,t} - nu_{n(j),t} + beta~^sigma_{j,a,b,t+1}          (Prop. 4)
#   (h, beta)   = support local (49) si [a:b] est ACTIF a l'etape t+1,
#                 (0, 0) sinon                                             (Prop. 4)
#   tau_{j,a,b} = h + tau~^sigma_{j,a,b,t+1}                               (36)
#
# La propagation tau~^sigma a lieu dans TOUS les cas, y compris sur un intervalle
# inactif : un intervalle qui demarre plus tard porte une valeur future non nulle.
# L'omettre rend la borne invalide.
#
# alpha n'est pas calcule par (37) : il est retro-calcule depuis la valeur de Bellman
# au point d'essai, de sorte que la coupe soit tendue par construction. Ce retro-calcul
# suppose la dualite forte, donc des multiplicateurs OPTIMAUX a l'etat d'essai sur les
# intervalles selectionnes (Prop. 4, cas gamma = 1) : c'est pourquoi le point de
# linearisation y_prev y vaut l'etat reel et non la trajectoire de reference.
function refine_bellman_function_extended(
    model::PolicyGraph{T},
    node::Node{T},
    selected_intervals::Vector{Tuple{Int,Int,Int}},
    outgoing_state_power::Vector{Float64},
    outgoing_state_gamma::Dict{Tuple{Int,Int,Int},Float64},
    outgoing_state_zeta::Vector{Float64},
    σ::Vector{Vector{Float64}},
    ν::Vector{Vector{Float64}},
    probability::Vector{Float64},
    objective_realizations::Vector{Float64},
) where {T}
    lock(node.lock)
    try
        instance = model.instance
        thermal_units = instance.Thermalunits
        horizon = instance.TimeHorizon
        arma = instance.arma
        t = node.index - 1
        cuts = model[node.index+1].interval_cuts
        nscen = length(objective_realizations)

        # tau~^sigma est lineaire en sigma : sa moyenne sur les aleas ne depend que de
        # ces poids. Un seul parcours des coupes par intervalle au lieu de nscen.
        σ_mean = [sum(probability[i] * σ[i][k] for i in 1:nscen) for k in 1:length(cuts)]

        # Coefficient de l'etat d'incertitude, eq. (10) puis (12) moyennees par (40b) :
        #     lambda   = S_eta' D_t nu_bar + delta~^sigma_{t+1}
        #     delta~_t = Psi' lambda
        # Tout est lineaire en (nu, sigma), donc la moyenne sur les aleas ne demande que
        # nu_bar et sigma_mean. Le facteur `force` est celui du bilan (1h) : a force = 0
        # la premiere part s'annule et, par recurrence depuis delta = 0 au dernier
        # etage, delta~ reste nul -- le cas deterministe est inchange.
        nz = MSUC.zeta_dimension(arma)
        nb = size(arma.S_eta, 1)
        ν_mean = [sum(probability[i] * ν[i][n] for i in 1:nscen) for n in 1:nb]
        λ = [sum(σ_mean[k] * cuts[k].delta[j] for k in 1:length(cuts)) for j in 1:nz]
        for j in 1:nz, n in 1:nb
            λ[j] += arma.S_eta[n, j] * instance.Demandbus[n][t+1] * ν_mean[n] * model.force
        end
        δ = [sum(arma.Psi[k, j] * λ[k] for k in 1:nz) for j in 1:nz]

        # ytilde_{j,a,b,t} au point d'essai. A la racine c'est l'etat initial (17) ;
        # ailleurs ytilde = u_{a,b,t} y_{j,t} (41g) et l'on renvoie y_{j,t} sans le
        # facteur u_{a,b,t} : licite car beta = 0 sur tout intervalle inactif.
        trial_output(unit, a) =
            node.index == 1 ? initial_interval_output(unit, a) :
            outgoing_state_power[unit.name]

        τ = Dict{Tuple{Int64,Int64,Int64},Float64}()
        β = Dict{Tuple{Int64,Int64,Int64},Float64}()
        for unit in thermal_units, (a, b) in unit.intervals
            key = (unit.name, a, b)
            τ_ab = sum(σ_mean[k] * cut.tau[key] for (k, cut) in enumerate(cuts))
            β_ab = 0.0
            if is_on_interval(a, b, t + 1)
                y_prev =
                    outgoing_state_gamma[key] >= 0.1 ? outgoing_state_power[unit.name] :
                    reference_output(unit, a, b, t, horizon)
                for i in 1:nscen
                    z =
                        unit.LinearTerm - ν[i][unit.Bus] +
                        sum(σ[i][k] * cut.beta[key] for (k, cut) in enumerate(cuts))
                    h, β_z = local_support(unit, a, b, t + 1, horizon, y_prev, z)
                    τ_ab += probability[i] * h
                    β_ab += probability[i] * β_z
                end
            end
            τ[key] = τ_ab
            β[key] = β_ab
        end

        value = sum(probability[i] * objective_realizations[i] for i in 1:nscen)
        α =
            value - sum(δ[j] * outgoing_state_zeta[j] for j in 1:nz; init = 0.0) - sum(
                outgoing_state_gamma[unit.name, a, b] * (
                    β[(unit.name, a, b)] * trial_output(unit, a) + τ[(unit.name, a, b)]
                ) for unit in thermal_units for (a, b) in unit.intervals
            )

        theta = node.bellman_function.global_theta.theta
        ζ_expr = sum(δ[j] * node.states_zeta[j].out for j in 1:nz; init = 0.0)
        cstr = if node.index == 1
            # ytilde_0 est une donnee (17) : la coupe ne depend plus que de gamma.
            @constraint(
                node.subproblem,
                theta >= α + ζ_expr + sum(
                    node.states_gamma[unit.name, a, b] * (
                        trial_output(unit, a) * β[(unit.name, a, b)] +
                        τ[(unit.name, a, b)]
                    ) for unit in thermal_units for (a, b) in unit.intervals
                )
            )
        else
            # ytilde_{j,a,b,t} = u_{a,b,t} y_{j,t} : la coupe s'ecrit sur la production
            # physique, au planning courant. C'est la ligne d'epigraphe (41g).
            @constraint(
                node.subproblem,
                theta >= α + ζ_expr + sum(
                    node.states_power[Symbol("power[$j]")].out * β[(j, a, b)] +
                    τ[(j, a, b)] for (j, a, b) in selected_intervals
                )
            )
        end
        push!(node.interval_cuts, IntervalCut(α, τ, β, δ, cstr))
    finally
        unlock(node.lock)
    end
    return
end

function _refine_bellman_function_no_lock(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    risk_measure::AbstractRiskMeasure,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
) where {T}
    # Sanity checks.
    @assert length(dual_variables) ==
            length(noise_supports) ==
            length(nominal_probability) ==
            length(objective_realizations)
    # Preliminaries that are common to all cut types.
    risk_adjusted_probability = similar(nominal_probability)
    offset = adjust_probability(
        risk_measure,
        risk_adjusted_probability,
        nominal_probability,
        noise_supports,
        objective_realizations,
        model.objective_sense == MOI.MIN_SENSE,
    )
    # The meat of the function.
    if bellman_function.cut_type == SINGLE_CUT
        return _add_average_cut(
            node,
            outgoing_state,
            risk_adjusted_probability,
            objective_realizations,
            dual_variables,
            offset,
        )
    else  # Add a multi-cut
        @assert bellman_function.cut_type == MULTI_CUT
        _add_locals_if_necessary(node, bellman_function, length(dual_variables))
        return _add_multi_cut(
            node,
            outgoing_state,
            risk_adjusted_probability,
            objective_realizations,
            dual_variables,
            offset,
        )
    end
end

function _add_average_cut(
    node::Node,
    outgoing_state::Dict{Symbol,Float64},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    offset::Float64,
)
    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables)
    # Calculate the expected intercept and dual variables with respect to the
    # risk-adjusted probability distribution.
    πᵏ = Dict(key => 0.0 for key in keys(outgoing_state))
    θᵏ = offset
    for i in 1:length(objective_realizations)
        p = risk_adjusted_probability[i]
        θᵏ += p * objective_realizations[i]
        for (key, dual) in dual_variables[i]
            πᵏ[key] += p * dual
        end
    end
    # Coupe moyenne ajoutee au sous-probleme.
    _add_cut(node.bellman_function.global_theta, θᵏ, πᵏ, outgoing_state)
    return (theta = θᵏ, pi = πᵏ, x = outgoing_state)
end

function _add_multi_cut(
    node::Node,
    outgoing_state::Dict{Symbol,Float64},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    offset::Float64,
)
    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables)
    bellman_function = node.bellman_function
    for i in 1:length(dual_variables)
        _add_cut(
            bellman_function.local_thetas[i],
            objective_realizations[i],
            dual_variables[i],
            outgoing_state,
        )
    end
    model = JuMP.owner_model(bellman_function.global_theta.theta)
    cut_expr = @expression(
        model,
        sum(
            risk_adjusted_probability[i] *
            bellman_function.local_thetas[i].theta for i in 1:N
        ) + offset
    )
    # TODO(odow): should we use `cut_expr` instead?
    ξ = copy(risk_adjusted_probability)
    if !(ξ in bellman_function.risk_set_cuts)
        push!(bellman_function.risk_set_cuts, ξ)
        if JuMP.objective_sense(model) == MOI.MIN_SENSE
            @constraint(model, bellman_function.global_theta.theta >= cut_expr)
        else
            @constraint(model, bellman_function.global_theta.theta <= cut_expr)
        end
    end
    return
end

# If we are adding a multi-cut for the first time, then the local θ variables
# won't have been added.
# TODO(odow): a way to set different bounds for each variable in the multi-cut.
function _add_locals_if_necessary(
    node::Node,
    bellman_function::BellmanFunction,
    N::Int,
)
    num_local_thetas = length(bellman_function.local_thetas)
    if num_local_thetas == N
        return # Do nothing. Already initialized.
    elseif num_local_thetas > 0
        error(
            "Expected $(N) local θ variables but there were " *
            "$(num_local_thetas).",
        )
    end
    global_theta = bellman_function.global_theta
    model = JuMP.owner_model(global_theta.theta)
    local_thetas = @variable(model, [1:N])
    if JuMP.has_lower_bound(global_theta.theta)
        JuMP.set_lower_bound.(
            local_thetas,
            JuMP.lower_bound(global_theta.theta),
        )
    end
    if JuMP.has_upper_bound(global_theta.theta)
        JuMP.set_upper_bound.(
            local_thetas,
            JuMP.upper_bound(global_theta.theta),
        )
    end
    for local_theta in local_thetas
        push!(
            bellman_function.local_thetas,
            ConvexApproximation(local_theta, global_theta.states),
        )
    end
    return
end

