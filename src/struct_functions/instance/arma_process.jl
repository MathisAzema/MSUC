"""
    ARMAProcess

Processus ARMA(p,q) de l'erreur relative de prevision, sous la forme d'etat (1g).

* `Psi`, `E` : matrices de (1g), de dimension `(p+q)` (elles contiennent les coefficients AR et MA
  du facteur scalaire commun a tous les noeuds).
* `S_eta` : extraction `eta_t = S_eta zeta_t`, de taille |N| × (p+q) (recopie de l'erreur
  du facteur sur chaque noeud).
* `support`, `probability` : discretisation a support fini de `xi_t` (Hypothese 1),
  une realisation est un vecteur de longueur 1.
* `zeta0` : etat initial, donnee du probleme.
"""
struct ARMAProcess
    Psi::Matrix{Float64}
    E::Matrix{Float64}
    S_eta::Matrix{Float64}
    support::Vector{Vector{Float64}}
    probability::Vector{Float64}
    zeta0::Vector{Float64}
end
