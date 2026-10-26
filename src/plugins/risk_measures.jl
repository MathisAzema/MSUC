#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

struct Expectation <: AbstractRiskMeasure end

function adjust_probability(
    ::Expectation,
    risk_adjusted_probability::Vector{Float64},
    original_probability::Vector{Float64},
    ::Vector,
    ::Vector{Float64},
    ::Bool,
)
    risk_adjusted_probability .= original_probability
    return 0.0
end

struct WorstCase <: AbstractRiskMeasure end

function adjust_probability(
    ::WorstCase,
    risk_adjusted_probability::Vector{Float64},
    original_probability::Vector{Float64},
    ::Vector,
    objective_realizations::Vector{Float64},
    is_minimization::Bool,
)
    risk_adjusted_probability .= 0.0
    worst_index = 1
    worst_observation = is_minimization ? -Inf : Inf
    for (index, (probability, observation)) in
        enumerate(zip(original_probability, objective_realizations))
        if probability > 0.0
            if (is_minimization && observation > worst_observation) ||
               (!is_minimization && observation < worst_observation)
                worst_index = index
                worst_observation = observation
            end
        end
    end
    risk_adjusted_probability[worst_index] = 1.0
    return 0.0
end

mutable struct Entropic <: AbstractRiskMeasure
    γ::Float64
end

function Base.show(io::IO, ent::Entropic)
    return print(io, "Entropic risk measure with γ = $(ent.γ)")
end

function adjust_probability(
    measure::Entropic,
    Q::Vector{Float64},
    p::Vector{Float64},
    ::Vector,
    X::Vector{Float64},
    is_min::Bool,
)
    if measure.γ == 0.0
        Q .= p
        return 0.0
    end
    γ = (is_min ? 1.0 : -1.0) * measure.γ
    y = p .* exp.(big.(γ .* X))
    Q .= y / sum(y)
    α = sum(
        qi * log(qi / pi) for (pi, qi) in zip(p, Q) if pi > 1e-14 && qi > 1e-14
    )
    return -α / γ
end
