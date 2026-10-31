#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Une coupe du noeud n minore la valeur future en fonction de l'etat sortant (y, zeta) ET du
# planning x : theta >= alpha + delta' zeta + beta' y + (terme en x). Sa forme depend de la
# base du planning : composantes (u, v, w) en 3-bin, intervalles [a:b] en formulation
# intervalle. `constraint_ref` est la ligne d'epigraphe de theta dans le sous-probleme.
abstract type AbstractCut end

mutable struct BinCut <: AbstractCut
    alpha::Float64
    tau::Vector{Float64}   # coefficients sur le vecteur plat x = (u, v, w)
    beta::Vector{Float64}  # coefficients sur y_j, une par unite
    delta::Vector{Float64} # coefficients sur zeta
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
end

mutable struct IntervalCut <: AbstractCut
    alpha::Float64
    tau::Vector{Float64}   # un coefficient par intervalle (identifiant de `IntervalIndex`)
    beta::Vector{Float64}
    delta::Vector{Float64}
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
end
