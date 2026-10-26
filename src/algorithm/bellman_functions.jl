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

struct InstanceFactory{T}
    args::Any
    kwargs::Any
    InstanceFactory{T}(args...; kwargs...) where {T} = new{T}(args, kwargs)
end

mutable struct BellmanFunction
    cut_type::CutType
    global_theta::ConvexApproximation
    local_thetas::Vector{ConvexApproximation}
    risk_set_cuts::Set{Vector{Float64}}
end

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
    model::PolicyGraph,
    node::Node,
)
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
    if node.next_node === nothing
        lower_bound = upper_bound = 0.0
    end
    Θᴳ = @variable(node.subproblem)

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

function refine_bellman_function_extended(
    model::PolicyGraph,
    node::Node,
    selected_intervals::Vector{Tuple{Int,Int,Int}},
    outgoing_state_power::Vector{Float64},
    outgoing_state_gamma::Dict{Tuple{Int,Int,Int},Float64},
    outgoing_state_zeta::Vector{Float64},
    σ::Vector{Vector{Float64}},
    ν::Vector{Vector{Float64}},
    probability::Vector{Float64},
    objective_realizations::Vector{Float64},
)
    instance = model.instance
    thermal_units = instance.Thermalunits
    horizon = instance.TimeHorizon
    arma = instance.arma
    t = node.index - 1
    cuts = model[node.index+1].interval_cuts
    nscen = length(objective_realizations)

    σ_mean = [sum(probability[i] * σ[i][k] for i in 1:nscen) for k in 1:length(cuts)]

    nz = MSUC.zeta_dimension(arma)
    nb = size(arma.S_eta, 1)
    ν_mean = [sum(probability[i] * ν[i][n] for i in 1:nscen) for n in 1:nb]
    λ = [sum(σ_mean[k] * cuts[k].delta[j] for k in 1:length(cuts)) for j in 1:nz]
    for j in 1:nz, n in 1:nb
        λ[j] += arma.S_eta[n, j] * instance.Demandbus[n][t+1] * ν_mean[n] * model.force
    end
    δ = [sum(arma.Psi[k, j] * λ[k] for k in 1:nz) for j in 1:nz]

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
        @constraint(
            node.subproblem,
            theta >= α + ζ_expr + sum(
                node.states_power[Symbol("power[$j]")].out * β[(j, a, b)] +
                τ[(j, a, b)] for (j, a, b) in selected_intervals
            )
        )
    end
    push!(node.interval_cuts, IntervalCut(α, τ, β, δ, cstr))
    return
end

function refine_bellman_function(
    model::PolicyGraph,
    node::Node,
    bellman_function::BellmanFunction,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
)
    @assert length(dual_variables) ==
            length(noise_supports) ==
            length(nominal_probability) ==
            length(objective_realizations)
    if bellman_function.cut_type == SINGLE_CUT
        return _add_average_cut(
            node,
            outgoing_state,
            nominal_probability,
            objective_realizations,
            dual_variables,
            0.0,
        )
    else
        @assert bellman_function.cut_type == MULTI_CUT
        _add_locals_if_necessary(node, bellman_function, length(dual_variables))
        return _add_multi_cut(
            node,
            outgoing_state,
            nominal_probability,
            objective_realizations,
            dual_variables,
            0.0,
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
    πᵏ = Dict(key => 0.0 for key in keys(outgoing_state))
    θᵏ = offset
    for i in 1:length(objective_realizations)
        p = risk_adjusted_probability[i]
        θᵏ += p * objective_realizations[i]
        for (key, dual) in dual_variables[i]
            πᵏ[key] += p * dual
        end
    end
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

function _add_locals_if_necessary(
    node::Node,
    bellman_function::BellmanFunction,
    N::Int,
)
    num_local_thetas = length(bellman_function.local_thetas)
    if num_local_thetas == N
        return
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
