struct ThermalUnit
    name::Int64
    Bus::Int64
    MinPower::Float64  
    MaxPower::Float64  
    DeltaRampUp::Float64  
    DeltaRampDown::Float64 
    StartUpCost::Float64
    StartDownCost::Float64  
    LinearTerm::Float64
    ConstTerm::Float64 
    InitialPower::Float64
    InitUpDownTime::Int64
    MinUpTime::Int64  
    MinDownTime::Int64
    intervals::Vector{Vector{Int64}}
    intervals_t::Vector{Vector{Vector{Int64}}}
end

struct Line
    b1::Int64
    b2::Int64 
    Fmax::Float64  
    B12::Float64 
end

"""
    IntervalIndex

Indexation plate des intervalles d'exploitation [a:b] de toutes les unites : l'identifiant `id`
numerote les intervalles unite par unite, dans l'ordre de `unit.intervals`. `unit[id]`, `a[id]`,
`b[id]` donnent l'intervalle, et `id[(j, a, b)]` l'identifiant d'un intervalle. Les coupes
intervalle stockent leurs coefficients dans des vecteurs indexes par cet identifiant.
"""
struct IntervalIndex
    unit::Vector{Int}
    a::Vector{Int}
    b::Vector{Int}
    id::Dict{Tuple{Int,Int,Int},Int}
end

function IntervalIndex(units::Vector{ThermalUnit})
    unit, a, b = Int[], Int[], Int[]
    id = Dict{Tuple{Int,Int,Int},Int}()
    for u in units, (aa, bb) in u.intervals
        push!(unit, u.name)
        push!(a, aa)
        push!(b, bb)
        id[(u.name, aa, bb)] = length(unit)
    end
    return IntervalIndex(unit, a, b, id)
end

struct Instance
    name::String
    TimeHorizon::Int64 
    N::Int64
    Thermalunits::Vector{ThermalUnit}
    intervals::IntervalIndex
    Lines::Dict{Tuple{Int64, Int64}, Line}
    Next::Vector{Vector{Int64}}
    Demandbus::Vector{Vector{Float64}}
    BusWind::Vector{Int64}
    WGscenario::Vector{Matrix{Float64}}
    Training_set::Vector{Vector{Int64}}
    # Processus ARMA de l'erreur de prevision (Sect. 2.1, eq. 1g)
    arma::ARMAProcess
    optimizer::Any
end
