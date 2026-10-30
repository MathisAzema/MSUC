"""
    ARMAProcess

Processus ARMA(p,q) de l'erreur relative de prevision, sous la forme d'etat (1g).

* `Phi`, `Theta` : coefficients AR et MA, appliques independamment a chaque noeud.
* `Psi`, `E`, `S_eta` : matrices de (1g) et de l'extraction `eta_t = S_eta zeta_t`.
* `support`, `probability` : discretisation a support fini de `xi_t` (Hypothese 1),
  une realisation est un vecteur de longueur |N|.
* `zeta0` : etat initial, donnee du probleme.
"""
struct ARMAProcess
    Phi::Vector{Float64}
    Theta::Vector{Float64}
    Psi::Matrix{Float64}
    E::Matrix{Float64}
    S_eta::Matrix{Float64}
    support::Vector{Vector{Float64}}
    probability::Vector{Float64}
    zeta0::Vector{Float64}
end
