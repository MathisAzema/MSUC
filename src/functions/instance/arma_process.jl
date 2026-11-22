# ARMA(p,q) process for the relative net-demand forecast error (Sect. 2.1):
#   eta_t = sum_i Phi_i eta_{t-i} + xi_t + sum_k Theta_k xi_{t-k}
# Markovian form with state zeta_t = (eta_t, .., eta_{t-p+1}, xi_t, .., xi_{t-q+1}):
#   zeta_t = Psi zeta_{t-1} + E xi_t,   eta_t = S_eta zeta_t      (eq. (1f); E is H in the paper)

# Calibrated ARMA(1,1) coefficients (Sect. 5.2, eq. (51)).
const ARMA_PHI = [0.67]
const ARMA_THETA = [0.117]

# Base innovation variance: xi_t ~ N(0, 2.9e-4 * sigma^2) (eq. (51)); `sigma` scales the standard deviation.
const ARMA_INNOVATION_VARIANCE = 2.9e-4

const ARMA_SUPPORT_SIZE = 5
const ARMA_DISCRETIZATION = :gauss_hermite

"""
    gauss_hermite(S) -> (nodes, weights)

`S`-point Gauss-Hermite quadrature for N(0,1) (Golub-Welsch).
"""
function gauss_hermite(S::Int)
    S <= 1 && return ([0.0], [1.0])
    J = LinearAlgebra.SymTridiagonal(zeros(S), [sqrt(k) for k in 1:(S-1)])
    F = LinearAlgebra.eigen(J)
    return collect(F.values), [abs2(F.vectors[1, i]) for i in 1:S]
end

# N(0,1) quantile via Acklam's rational approximation (relative error < 1.2e-9);
# avoids a SpecialFunctions dependency.
function standard_normal_quantile(p::Float64)
    @assert 0 < p < 1
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= phigh
        q = p - 0.5; r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6]) * q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

# N(0,1) discretized into `S` equiprobable strata: weight 1/S, node = conditional mean of the
# stratum, m_i = S (phi(z_{i-1}) - phi(z_i)) with z_i = Phi^{-1}(i/S). The mean is exact but the
# variance is underestimated (see `match_variance`).
function equiprobable_normal(S::Int)
    S <= 1 && return ([0.0], [1.0])
    φ(z) = exp(-z^2 / 2) / sqrt(2π)
    z = [standard_normal_quantile(i / S) for i in 1:(S-1)]
    dens = vcat(0.0, φ.(z), 0.0)
    return [S * (dens[i] - dens[i+1]) for i in 1:S], fill(1 / S, S)
end

"""
    innovation_support(S, sd; discretization, match_variance) -> (nodes, weights)

Finite support of the innovation with standard deviation `sd`, using `:gauss_hermite` or
`:equiprobable` weights. `match_variance` rescales the nodes so the discrete variance is exactly
`sd^2` (the equiprobable scheme underestimates it, e.g. 0.897 sd^2 at S=5).
"""
function innovation_support(
    S::Int,
    sd::Float64;
    discretization::Symbol = ARMA_DISCRETIZATION,
    match_variance::Bool = true,
)
    nodes, weights = if discretization === :gauss_hermite
        gauss_hermite(S)
    elseif discretization === :equiprobable
        equiprobable_normal(S)
    else
        error("unknown discretization: $discretization (:gauss_hermite or :equiprobable)")
    end
    if match_variance && S > 1
        m2 = sum(weights .* nodes .^ 2)
        m2 > 0 && (nodes ./= sqrt(m2))
    end
    return sd .* nodes, weights
end

"""
    ARMAProcess(nbus; sigma, Phi, Theta, S, discretization)

Build the process and its companion matrices for `nbus` buses.

A single scalar ARMA(p,q) factor drives the demand of all buses (the relative error is the same
everywhere, as in Zou, Ahmed & Sun, 2018), so `zeta` has dimension `p + q` and each stage has
exactly `S` noise outcomes. `sigma` scales the innovation standard deviation (variance
`2.9e-4 * sigma^2`); `sigma = 0` gives the deterministic problem, and `sigma = c` has the same
effect as `force = c` in `PolicyGraph`.
"""
function ARMAProcess(
    nbus::Int;
    sigma::Float64 = 1.0,
    Phi::Vector{Float64} = ARMA_PHI,
    Theta::Vector{Float64} = ARMA_THETA,
    variance::Float64 = ARMA_INNOVATION_VARIANCE * sigma^2,
    S::Int = ARMA_SUPPORT_SIZE,
    discretization::Symbol = ARMA_DISCRETIZATION,
    match_variance::Bool = true,
)
    p, q = length(Phi), length(Theta)
    n = p + q

    Psi = zeros(n, n)
    Psi[1, 1:p] .= Phi
    Psi[1, (p+1):(p+q)] .= Theta
    for i in 2:p
        Psi[i, i-1] = 1.0
    end
    for k in 2:q
        Psi[p+k, p+k-1] = 1.0
    end
    # The innovation enters eta_t and is stored in component p+1.
    E = zeros(n, 1)
    E[1, 1] = 1.0
    q >= 1 && (E[p+1, 1] = 1.0)

    S_eta = zeros(nbus, n)
    S_eta[:, 1] .= 1.0

    nodes, weights = innovation_support(
        S,
        sqrt(variance);
        discretization = discretization,
        match_variance = match_variance,
    )
    support = [[z] for z in nodes]
    return ARMAProcess(Psi, E, S_eta, support, weights, zeros(n))
end

zeta_dimension(a::ARMAProcess) = size(a.Psi, 1)
