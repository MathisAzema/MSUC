"""
    parse_nc4(name_instance, optimizer, T = 24; sigma, S, discretization)

Instance SMS++ de la T-Ramp (`.nc4`) : un seul noeud, sans reseau, demande nette nominale lue dans
le fichier. L'incertitude est le processus ARMA de l'erreur relative (`ARMAProcess`).
"""
function parse_nc4(name_instance, optimizer, T=24; sigma::Float64=1.0, S::Int=ARMA_SUPPORT_SIZE, discretization::Symbol=ARMA_DISCRETIZATION)
    data=Dataset(name_instance)
    Blocks=keys(data.group)
    data_block=data.group[Blocks[1]]
    TimeHorizon= T
    demand=data_block["ActivePowerDemand"].var[:]
    N=size(keys(data_block.group))[1]-1
    Thermal_units=Vector{ThermalUnit}(undef, N)
    for unit in data_block.group
        if occursin("Block", first(unit))
            unit_name=parse(Int64, split(first(unit), "_")[end])+1
            type=last(unit).attrib["type"]
            if type == "ThermalUnitBlock" 
                Bus = 1
                MinPower=last(unit)["MinPower"].var[1]
                MaxPower =last(unit)["MaxPower"].var[1] 
                DeltaRampUp  =last(unit)["DeltaRampUp"].var[1]
                DeltaRampDown  =last(unit)["DeltaRampDown"].var[1]
                StartUpCost=last(unit)["StartUpCost"].var[1]
                StartDownCost=0.0 
                LinearTerm=last(unit)["LinearTerm"].var[1]
                ConstTerm=last(unit)["ConstTerm"].var[1]
                InitialPower=last(unit)["InitialPower"].var[1]  
                InitUpDownTime =last(unit)["InitUpDownTime"].var[1] 
                MinUpTime=last(unit)["MinUpTime"].var[1]  
                MinDownTime=last(unit)["MinDownTime"].var[1]
                intervals=set_intervals(Int64(MinUpTime), InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)
                unit=ThermalUnit(unit_name, Bus, MinPower, MaxPower, DeltaRampUp, DeltaRampDown, StartUpCost, StartDownCost, LinearTerm, ConstTerm, InitialPower, InitUpDownTime, Int64(MinUpTime), Int64(MinDownTime), intervals)
                Thermal_units[unit_name]=unit
            end
        end
    end
    Lines=Dict{Tuple{Int64,Int64},Line}()
    Next=[Int64[]]
    Demandbus=[demand]
    return Instance(name_instance, TimeHorizon, N, Thermal_units, IntervalIndex(Thermal_units), Lines, Next, Demandbus, ARMAProcess(length(Demandbus); sigma=sigma, S=S, discretization=discretization), optimizer)
end

"""
    parse_IEEE(folder, optimizer, T = 24; sigma, S, discretization)

Instance IEEE 118 noeuds (`Data/IEEE/<folder>`, par exemple `"118_syst"`) : 54 groupes, reseau en
flux DC (chaque ligne dans les deux sens), horizon de `T <= 24` heures.

Demande nette nominale du noeud b a l'heure t : charge maximale du noeud × profil horaire (l'eolien
n'est pas modelise). L'incertitude est un unique processus ARMA de l'erreur RELATIVE, commun a tous
les noeuds : le ratio prevision/realise est le meme partout, c'est-a-dire une charge nette de reseau
repartie sur les noeuds au prorata des charges, comme dans l'etude IEEE 118 de Zou, Ahmed & Sun
(2018). L'etat d'incertitude `zeta` est donc de dimension 2 (ARMA(1,1)), et non 2 × 118.
"""
function parse_IEEE(
    folder,
    optimizer,
    T = 24;
    sigma::Float64 = 1.0,
    S::Int = ARMA_SUPPORT_SIZE,
    discretization::Symbol = ARMA_DISCRETIZATION,
)
    dir = joinpath(@__DIR__, "..", "..", "..", "Data", "IEEE", folder)
    generators = CSV.read(joinpath(dir, "generators.csv"), DataFrame; header = false)
    N = Int(generators[end, 1])
    units = Vector{ThermalUnit}(undef, N)
    for i in 1:N
        Bus, ConstTerm, LinearTerm = generators[i, 2], generators[i, 3], generators[i, 4]
        MaxPower, MinPower = generators[i, 5], generators[i, 6]
        DeltaRampUp = DeltaRampDown = generators[i, 7]
        StartUpCost = StartDownCost = generators[i, 10]
        MinUpTime = MinDownTime = generators[i, 15]
        InitialPower = generators[i, 12]
        InitUpDownTime =
            MinUpTime * (InitialPower >= 1e-3) - MinUpTime * (InitialPower <= 1e-3)
        intervals = set_intervals(Int64(MinUpTime), InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)
        units[i] = ThermalUnit(
            i, Bus, MinPower, MaxPower, DeltaRampUp, DeltaRampDown, StartUpCost, StartDownCost,
            LinearTerm, ConstTerm, InitialPower, Int64(InitUpDownTime), Int64(MinUpTime),
            Int64(MinDownTime), intervals,
        )
    end

    max_load = CSV.read(joinpath(dir, "maximum_load.csv"), DataFrame; header = false)
    nbus = Int(max_load[end, 1])
    profile =
        CSV.read(joinpath(dir, "load_distribution_profile.csv"), DataFrame; header = false)[:, 2] / 100
    T <= length(profile) ||
        error("le profil de charge couvre $(length(profile)) heures, T = $T est trop grand")
    Demandbus = [max_load[b, 2] * profile[1:T] for b in 1:nbus]

    lines = CSV.read(joinpath(dir, "lines.csv"), DataFrame; header = false)
    Lines = Dict{Tuple{Int64,Int64},Line}()
    Next = [Int64[] for _ in 1:nbus]
    for i in 1:Int(lines[end, 1])
        b1, b2, reactance, capacity = Int(lines[i, 2]), Int(lines[i, 3]), lines[i, 4], lines[i, 5]
        Lines[(b1, b2)] = Line(b1, b2, capacity, 1 / reactance)
        Lines[(b2, b1)] = Line(b2, b1, capacity, 1 / reactance)
        push!(Next[b1], b2)
        push!(Next[b2], b1)
    end

    arma = ARMAProcess(nbus; sigma = sigma, S = S, discretization = discretization)
    return Instance(
        "IEEE$N", T, N, units, IntervalIndex(units), Lines, Next, Demandbus, arma, optimizer,
    )
end
