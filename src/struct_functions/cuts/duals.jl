#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    StageDuals

Multiplicateurs d'un sous-probleme d'etape, dans les notations du papier :
`σ` duaux des lignes d'epigraphe, `ν` du bilan par noeud, `β` et `δ` des etats entrants y et
zeta, `τ` des parametres du planning (3-bin ; vide en formulation intervalle).
"""
struct StageDuals
    σ::Vector{Float64}
    ν::Vector{Float64}
    β::Vector{Float64}
    δ::Vector{Float64}
    τ::Vector{Float64}
end
