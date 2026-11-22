"""
    ARMAProcess

ARMA(p,q) process of the relative forecast error in state form (1f):
`zeta_t = Psi zeta_{t-1} + E xi_t` (`E` is H in the paper), `eta_t = S_eta zeta_t` (`S_eta` copies the error
to every node).
`support`/`probability` discretize the innovation `xi_t`; `zeta0` is the initial state.
"""
struct ARMAProcess
    Psi::Matrix{Float64}
    E::Matrix{Float64}
    S_eta::Matrix{Float64}
    support::Vector{Vector{Float64}}
    probability::Vector{Float64}
    zeta0::Vector{Float64}
end
