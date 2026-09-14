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

struct Instance
    name::String
    TimeHorizon::Int64 
    N::Int64
    Thermalunits::Vector{ThermalUnit}
    Lines::Dict{Tuple{Int64, Int64}, Line}
    Next::Vector{Vector{Int64}}
    Demandbus::Vector{Vector{Float64}}
    BusWind::Vector{Int64}
    WGscenario::Vector{Matrix{Float64}}
    Training_set::Vector{Vector{Int64}}
    optimizer::Any
end