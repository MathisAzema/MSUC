#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    StageDuals

Duals of a stage subproblem: `σ` epigraph rows, `ν` demand balance, `β`/`δ` incoming states
y/zeta, `τ` schedule parameters. `β` and `τ` are empty in the interval formulation.
"""
struct StageDuals
    σ::Vector{Float64}
    ν::Vector{Float64}
    β::Vector{Float64}
    δ::Vector{Float64}
    τ::Vector{Float64}
end
