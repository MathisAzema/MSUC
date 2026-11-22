#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Cut on the future value: theta >= alpha + delta' zeta + beta' y + (term in x). `tau` is indexed by
# (u, v, w) components for BinCut and by interval id for IntervalCut. `constraint_ref` is the epigraph row.
abstract type AbstractCut end

mutable struct BinCut <: AbstractCut
    alpha::Float64
    tau::Vector{Float64}
    beta::Vector{Float64}
    delta::Vector{Float64}
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
    applied::Vector{Float64}   # y-slopes already written in the epigraph row (empty: none)
end

mutable struct IntervalCut <: AbstractCut
    alpha::Float64
    tau::Vector{Float64}
    beta::Vector{Float64}
    delta::Vector{Float64}
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
    applied::Vector{Float64}
end

BinCut(alpha, tau, beta, delta, constraint_ref) =
    BinCut(alpha, tau, beta, delta, constraint_ref, Float64[])

IntervalCut(alpha, tau, beta, delta, constraint_ref) =
    IntervalCut(alpha, tau, beta, delta, constraint_ref, Float64[])
