"""
    read_data_forecast(file)

Lit un fichier CSV de scenarios de demande nette (sans en-tete).
"""
function read_data_forecast(file)
    return CSV.read(file, DataFrame; delim=",", header=0)
end

function parse_nc4(name_instance, optimizer, T=24; sigma::Float64=1.0, S::Int=ARMA_SUPPORT_SIZE, discretization::Symbol=ARMA_DISCRETIZATION)
    """
    Parse the SMS++ instances
    """
    data=Dataset(name_instance)
    Blocks=keys(data.group)
    data_block=data.group[Blocks[1]]
    # TimeHorizon= data_block.dim["TimeHorizon"]
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
                intervals_t=[set_intervals_t(t, Int64(MinUpTime), InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T) for t in 1:T]

                unit=ThermalUnit(unit_name, Bus, MinPower, MaxPower, DeltaRampUp, DeltaRampDown, StartUpCost, StartDownCost, LinearTerm, ConstTerm, InitialPower, InitUpDownTime, Int64(MinUpTime), Int64(MinDownTime), intervals, intervals_t)
                Thermal_units[unit_name]=unit
            end
        end
    end
    Buses=1:1
    Next=[[]]
    Lines=Dict()
    Demandbus=[demand]
    BusWind=[1]

    training_file = "Data/Uncertainty/training_set.csv"
    table = read_data_forecast(training_file)
    Nb_total_scenario_training=table[end,end]
    Smax=100
    Num_batch=Int(floor(Nb_total_scenario_training/Smax))
    WGscenariob=[]
    for batch in 1:Num_batch
        for day in 1:Smax
            push!(WGscenariob,-table[(batch-1)*Smax*24+(day-1)*24+1:(batch-1)*Smax*24+day*24,1].*demand)
        end
    end
    Training_set=[[(batch-1)*Smax+day for day in 1:Smax] for batch in 1:Num_batch]
    Nb_total_scenario_training=Num_batch*Smax
    
    test_file = "Data/Uncertainty/test_set.csv"
    table = read_data_forecast(test_file)
    Nb_total_scenario_test=table[end,end]
    for day in 1:Nb_total_scenario_test
        push!(WGscenariob,-table[(day-1)*24+1:day*24,1].*demand)
    end
    push!(Training_set, [day for day in Nb_total_scenario_training+1:Nb_total_scenario_training+Nb_total_scenario_test])
    WGscenario=[hcat(WGscenariob...)]
    return Instance(name_instance, TimeHorizon, N, Thermal_units, IntervalIndex(Thermal_units), Lines, Next, Demandbus, BusWind, WGscenario, Training_set, ARMAProcess(length(Demandbus); sigma=sigma, S=S, discretization=discretization), optimizer)
end
