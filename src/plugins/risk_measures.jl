#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# ========================== The Expectation Operator ======================== #

"""
    Expectation()

The Expectation risk measure.

This risk measure is identical to taking the expectation with respect to the
nominal distribution.

## Example

```jldoctest
julia> risk_adjusted_probability = zeros(4);

julia> MSUC.adjust_probability(
           MSUC.Expectation(),
           risk_adjusted_probability,
           [0.1, 0.2, 0.3, 0.4],  # nominal_probability,
           MSUC.Noise.([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),  # noise_supports,
           [5.0, 4.0, 6.0, 2.0],  # cost_realizations,
           true,                  # is_minimization
       )
0.0

julia> risk_adjusted_probability
4-element Vector{Float64}:
 0.1
 0.2
 0.3
 0.4
```
"""
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

# ========================== The Worst Case Operator ========================= #

"""
    WorstCase()

The worst-case risk measure.

This risk measure places all of the probability weight on the worst outcome.

## Example

```jldoctest
julia> risk_adjusted_probability = zeros(4);

julia> MSUC.adjust_probability(
           MSUC.WorstCase(),
           risk_adjusted_probability,
           [0.1, 0.2, 0.3, 0.4],  # nominal_probability,
           MSUC.Noise.([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),  # noise_supports,
           [5.0, 4.0, 6.0, 2.0],  # cost_realizations,
           true,                  # is_minimization
       )
0.0

julia> risk_adjusted_probability
4-element Vector{Float64}:
 0.0
 0.0
 1.0
 0.0
```
"""
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

# ================================= Entropic ============================== #

"""
    Entropic(γ::Float64)

The entropic risk measure as described by:

    Dowson, O., Morton, D.P. & Pagnoncelli, B.K. Incorporating convex risk
    measures into multistage stochastic programming algorithms. Annals of
    Operations Research (2022). [doi](https://doi.org/10.1007/s10479-022-04977-w).

As γ increases, the measure becomes more risk-averse.

## Example

```jldoctest
julia> risk_adjusted_probability = zeros(4);

julia> MSUC.adjust_probability(
           MSUC.Entropic(0.1),
           risk_adjusted_probability,
           [0.1, 0.2, 0.3, 0.4],  # nominal_probability,
           MSUC.Noise.([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),  # noise_supports,
           [5.0, 4.0, 6.0, 2.0],  # cost_realizations,
           true,                  # is_minimization
       )
-0.14333892665462006

julia> risk_adjusted_probability
4-element Vector{Float64}:
 0.1100296362588547
 0.19911786395979578
 0.3648046623591841
 0.3260478374221655

julia> MSUC.adjust_probability(
           MSUC.Entropic(1.0),
           risk_adjusted_probability,
           [0.1, 0.2, 0.3, 0.4],  # nominal_probability,
           MSUC.Noise.([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),  # noise_supports,
           [5.0, 4.0, 6.0, 2.0],  # cost_realizations,
           true,                  # is_minimization
       )
-0.12038063114659443

julia> risk_adjusted_probability
4-element Vector{Float64}:
 0.09911045746726178
 0.07292139941460454
 0.8082304666305623
 0.019737676487571337

julia> MSUC.adjust_probability(
           MSUC.Entropic(10.0),
           risk_adjusted_probability,
           [0.1, 0.2, 0.3, 0.4],  # nominal_probability,
           MSUC.Noise.([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),  # noise_supports,
           [5.0, 4.0, 6.0, 2.0],  # cost_realizations,
           true,                  # is_minimization
       )
-0.12038063114659443

julia> risk_adjusted_probability
4-element Vector{Float64}:
 1.5133080886430772e-5
 1.374081618667918e-9
 0.999984865545032
 5.664386611687232e-18
```
"""
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
    if measure.γ == 0.0  # Special case of entropic: if γ = 0, F[X] = E[X].
        Q .= p
        return 0.0
    end
    # Handle maximization problems by negating γ. Usually we would negate X, but
    # with convex RM's, we also have to negate the α(q) term, so it's easier to
    # just negate γ.
    γ = (is_min ? 1.0 : -1.0) * measure.γ
    # Use `BigFloat` to avoid overflow that occurs when calculating `exp(x)`.
    y = p .* exp.(big.(γ .* X))
    Q .= y / sum(y)
    α = sum(
        # To avoid numerical issues with the log, skip elements that are `≈ 0`.
        qi * log(qi / pi) for (pi, qi) in zip(p, Q) if pi > 1e-14 && qi > 1e-14
    )
    return -α / γ
end
