function subproblem_builder_UC_def(instance::MSUC.Instance, force::Float64, subproblem::Model, node::Int)
    # State variables
    N=instance.N
    T= instance.TimeHorizon
    thermal_units=values(instance.Thermalunits)
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    BusWind=instance.BusWind
    NumWindfarms=length(BusWind)

    @variable(subproblem, 0.0 <= power[i = 1:N] <= thermal_units[i].MaxPower, MSUC.State, initial_value = thermal_units[i].InitialPower)
    
    @variable(subproblem, 0.0 <= is_on[i = 1:N, t in 0:T] <= 1.0, MSUC.State, initial_value = 0.0) #initial value fictive
    @variable(subproblem, 0.0 <= start_up[i = 1:N, t in 1:T] <= 1.0, MSUC.State, initial_value = 0.0)
    @variable(subproblem, 0.0 <= start_down[i = 1:N, t in 1:T] <= 1.0, MSUC.State, initial_value = 0.0)

    @variable(subproblem, gamma[i in 1:N, (a,b) in instance.Thermalunits[i].intervals]>=0)
    @constraint(subproblem, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a<=t<b)==is_on[unit.name,t].out)
    @constraint(subproblem, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a==t)==start_up[unit.name,t].out)
    @constraint(subproblem, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if b==t)==start_down[unit.name,t].out)

    cstr = JuMP.ConstraintRef[]
    
    if node == 1
        for i in 1:N
            JuMP.set_binary.(is_on[i, 0].out)
            for t in 1:T
                JuMP.set_binary.(is_on[i, t].out)
                JuMP.set_binary.(start_up[i, t].out)
                JuMP.set_binary.(start_down[i, t].out)
            end
        end

        @constraint(subproblem,  [unit in thermal_units, t in 1:T], is_on[unit.name, t].out-is_on[unit.name, t-1].out==start_up[unit.name, t].out-start_down[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T], start_up[unit.name, t].out<=is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T], start_down[unit.name, t].out<=1-is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_up[unit.name, τ].out for τ in max(1, t-unit.MinUpTime+1):t)+1*(t<unit.MinUpTime-unit.InitUpDownTime+1)*(unit.InitUpDownTime>0) <= is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_down[unit.name, τ].out for τ in max(1, t-unit.MinDownTime+1):t)+1*(t<unit.MinDownTime+unit.InitUpDownTime+1)*(unit.InitUpDownTime<0) <= 1-is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_up[unit.name, τ].out for τ in max(1, t-unit.MinUpTime+1):t) <= is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_down[unit.name, τ].out for τ in max(1, t-unit.MinDownTime+1):t) <= 1-is_on[unit.name, t].out)

        @constraint(subproblem,  [unit in thermal_units; unit.InitUpDownTime!=nothing], is_on[unit.name, 0].out==(unit.InitUpDownTime>=0))

        @constraint(subproblem,  [unit in thermal_units; unit.InitUpDownTime==nothing], is_on[unit.name, 0].out==0)

        for unit in thermal_units
            if (unit.InitialPower-unit.MinPower)/unit.DeltaRampDown <0
                limit=-1
            else
                limit=Int64(ceil((unit.InitialPower-unit.MinPower)/unit.DeltaRampDown))
            end
            for t in 0:limit
                @constraint(subproblem, is_on[unit.name, t].out==1)
            end
        end

        @stageobjective(subproblem, sum(unit.ConstTerm*is_on[unit.name, t].out+unit.StartUpCost*start_up[unit.name, t].out+unit.StartDownCost*start_down[unit.name, t].out for unit in thermal_units for t in 1:T))

        @constraint(subproblem,  [i in 1:N], power[i].out == power[i].in)

    else

        # Control variables
        @variables(subproblem, begin
            power_shedding[b in Buses] >= 0
            power_curtailement[b in Buses] >= 0
            θ[b in Buses] 
            flow[b in Buses, bp in Next[b]]
        end)

        @constraint(subproblem,  [i in 1:N, time in 0:T], is_on[i, time].out == is_on[i, time].in)
        @constraint(subproblem,  [i in 1:N, time in 1:T], start_up[i, time].out == start_up[i, time].in)
        @constraint(subproblem,  [i in 1:N, time in 1:T], start_down[i, time].out == start_down[i, time].in)

        t=node - 1

        # Random variables
        @variable(subproblem, error_forecast[k in 1:NumWindfarms])
        Ω = [[0.0 for b in BusWind]]
        P = [1.0]
        MSUC.parameterize(subproblem, Ω, P) do ω
            for k in 1:NumWindfarms
                JuMP.fix(error_forecast[k], ω[k])
            end
        end
        
        @constraint(subproblem, [i = 1: N], power[i].out <= sum(min(thermal_units[i].MaxPower, thermal_units[i].MinPower + thermal_units[i].DeltaRampDown * (b-t-1)) * gamma[i, [a,b]] for (a,b) in thermal_units[i].intervals_t[t] if b <= 24) + sum(thermal_units[i].MaxPower * gamma[i, [a,b]] for (a,b) in thermal_units[i].intervals_t[t] if b > 24))


        # Transition function and constraints
        @constraints(
            subproblem,
            begin
                [i = 1: N], power[i].out <= thermal_units[i].MaxPower * is_on[i, t].out
                [i = 1: N], power[i].out >= thermal_units[i].MinPower * is_on[i, t].out
                [i = 1:N], power[i].out - power[i].in <= (-thermal_units[i].DeltaRampUp)*start_up[i,t].out + (thermal_units[i].MinPower + thermal_units[i].DeltaRampUp)*is_on[i, t].out - (thermal_units[i].MinPower)*is_on[i, t-1].out
                [i = 1: N], power[i].in - power[i].out <= (-thermal_units[i].DeltaRampDown)*start_down[i,t].out + (thermal_units[i].MinPower + thermal_units[i].DeltaRampDown)*is_on[i, t-1].out - (thermal_units[i].MinPower)*is_on[i, t].out
                [line in Lines], flow[line.b1,line.b2] <= line.Fmax
                [line in Lines], flow[line.b1,line.b2] >= -line.Fmax
                [line in Lines], flow[line.b1,line.b2] == line.B12*(θ[line.b1]-θ[line.b2])
            end
        )
        cstr = [@constraint(subproblem,sum(power[unit.name].out for unit in thermal_units if unit.Bus==b) - power_curtailement[b] + power_shedding[b] == instance.Demandbus[b][t]*(1+force*sum(error_forecast[w] for w in 1:NumWindfarms if BusWind[w]==b))+sum(flow[b,bp] for bp in Next[b])) for b in Buses]
        # Stage-objective
        @stageobjective(subproblem, sum(unit.LinearTerm*power[unit.name].out for unit in thermal_units)+sum(MSUC.SHEDDING_COST*power_shedding[b]+MSUC.CURTAILEMENT_COST*power_curtailement[b] for b in Buses))

        return cstr
        end
    return cstr
end

function subproblem_builder_UC_extended(instance::MSUC.Instance, force::Float64, subproblem::Model, node::Int)
    # State variables
    N=instance.N
    T= instance.TimeHorizon
    thermal_units=values(instance.Thermalunits)
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    BusWind=instance.BusWind
    NumWindfarms=length(BusWind)

    @variable(subproblem, 0.0 <= power[i = 1:N] <= thermal_units[i].MaxPower, MSUC.State, initial_value = thermal_units[i].InitialPower)
    
    @variable(subproblem, 0.0 <= is_on[i = 1:N, t in 0:T] <= 1.0, MSUC.State, initial_value = 0.0) #initial value fictive
    @variable(subproblem, 0.0 <= start_up[i = 1:N, t in 1:T] <= 1.0, MSUC.State, initial_value = 0.0)
    @variable(subproblem, 0.0 <= start_down[i = 1:N, t in 1:T] <= 1.0, MSUC.State, initial_value = 0.0)

    cstr = JuMP.ConstraintRef[]

    z_var = Dict{Tuple{Int, Int,Int},JuMP.VariableRef}()
    gamma_var = Dict{Tuple{Int, Int,Int},JuMP.VariableRef}()
    for i in 1:N
        for (a,b) in instance.Thermalunits[i].intervals
            z_var[i,a,b] = @variable(subproblem, lower_bound = 0.0, base_name = "z_$(i)_$(a)_$(b)")
            gamma_var[i,a,b] = @variable(subproblem, lower_bound = 0.0, base_name = "gamma_$(i)_$(a)_$(b)")
        end
    end
    @constraint(subproblem, [unit in thermal_units, t in 1:T], sum(gamma_var[unit.name, a,b] for (a,b) in unit.intervals if a<=t<b)==is_on[unit.name,t].out)
    @constraint(subproblem, [unit in thermal_units, t in 1:T], sum(gamma_var[unit.name, a,b] for (a,b) in unit.intervals if a==t)==start_up[unit.name,t].out)
    @constraint(subproblem, [unit in thermal_units, t in 1:T], sum(gamma_var[unit.name, a,b] for (a,b) in unit.intervals if b==t)==start_down[unit.name,t].out)

    if node == 1
        for i in 1:N
            JuMP.set_binary.(is_on[i, 0].out)
            for t in 1:T
                JuMP.set_binary.(is_on[i, t].out)
                JuMP.set_binary.(start_up[i, t].out)
                JuMP.set_binary.(start_down[i, t].out)
            end
        end

        @constraint(subproblem, [unit in thermal_units, (a,b) in unit.intervals], z_var[unit.name,a,b] == unit.InitialPower * gamma_var[unit.name,a,b])

        @constraint(subproblem,  [unit in thermal_units, t in 1:T], is_on[unit.name, t].out-is_on[unit.name, t-1].out==start_up[unit.name, t].out-start_down[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T], start_up[unit.name, t].out<=is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T], start_down[unit.name, t].out<=1-is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_up[unit.name, τ].out for τ in max(1, t-unit.MinUpTime+1):t)+1*(t<unit.MinUpTime-unit.InitUpDownTime+1)*(unit.InitUpDownTime>0) <= is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_down[unit.name, τ].out for τ in max(1, t-unit.MinDownTime+1):t)+1*(t<unit.MinDownTime+unit.InitUpDownTime+1)*(unit.InitUpDownTime<0) <= 1-is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_up[unit.name, τ].out for τ in max(1, t-unit.MinUpTime+1):t) <= is_on[unit.name, t].out)
        @constraint(subproblem,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_down[unit.name, τ].out for τ in max(1, t-unit.MinDownTime+1):t) <= 1-is_on[unit.name, t].out)

        @constraint(subproblem,  [unit in thermal_units; unit.InitUpDownTime!=nothing], is_on[unit.name, 0].out==(unit.InitUpDownTime>=0))

        @constraint(subproblem,  [unit in thermal_units; unit.InitUpDownTime==nothing], is_on[unit.name, 0].out==0)

        for unit in thermal_units
            if (unit.InitialPower-unit.MinPower)/unit.DeltaRampDown <0
                limit=-1
            else
                limit=Int64(ceil((unit.InitialPower-unit.MinPower)/unit.DeltaRampDown))
            end
            for t in 0:limit
                @constraint(subproblem, is_on[unit.name, t].out==1)
            end
        end

        @stageobjective(subproblem, sum(unit.ConstTerm*is_on[unit.name, t].out+unit.StartUpCost*start_up[unit.name, t].out+unit.StartDownCost*start_down[unit.name, t].out for unit in thermal_units for t in 1:T))

        @constraint(subproblem,  [i in 1:N], power[i].out == power[i].in)

    else

        # Control variables
        @variables(subproblem, begin
            power_shedding[b in Buses] >= 0
            power_curtailement[b in Buses] >= 0
            θ[b in Buses] 
            flow[b in Buses, bp in Next[b]]
        end)

        @constraint(subproblem,  [i in 1:N, time in 0:T], is_on[i, time].out == is_on[i, time].in)
        @constraint(subproblem,  [i in 1:N, time in 1:T], start_up[i, time].out == start_up[i, time].in)
        @constraint(subproblem,  [i in 1:N, time in 1:T], start_down[i, time].out == start_down[i, time].in)

        t=node - 1

        @constraint(subproblem, [unit in thermal_units, (a,b) in unit.intervals_t_extended[t]], z_var[unit.name,a,b] <= power[unit.name].out)
        @constraint(subproblem, [unit in thermal_units, (a,b) in unit.intervals_t_extended[t]], z_var[unit.name,a,b] <= unit.MaxPower * gamma_var[unit.name,a,b])
        @constraint(subproblem, [unit in thermal_units, (a,b) in unit.intervals_t_extended[t]], z_var[unit.name,a,b] >= power[unit.name].out - unit.MaxPower * (1 - gamma_var[unit.name,a,b]))

        # Random variables
        @variable(subproblem, error_forecast[k in 1:NumWindfarms])
        Ω = [[0.0 for b in BusWind]]
        P = [1.0]
        MSUC.parameterize(subproblem, Ω, P) do ω
            for k in 1:NumWindfarms
                JuMP.fix(error_forecast[k], ω[k])
            end
        end

        @constraint(subproblem, [i = 1: N], power[i].out <= sum(min(thermal_units[i].MaxPower, thermal_units[i].MinPower + thermal_units[i].DeltaRampDown * (b-t-1)) * gamma_var[i, a,b] for (a,b) in thermal_units[i].intervals_t[t] if b <= 24) + sum(thermal_units[i].MaxPower * gamma_var[i, a,b] for (a,b) in thermal_units[i].intervals_t[t] if b > 24))


        # Transition function and constraints
        @constraints(
            subproblem,
            begin
                [i = 1: N], power[i].out <= thermal_units[i].MaxPower * is_on[i, t].out
                [i = 1: N], power[i].out >= thermal_units[i].MinPower * is_on[i, t].out
                [i = 1:N], power[i].out - power[i].in <= (-thermal_units[i].DeltaRampUp)*start_up[i,t].out + (thermal_units[i].MinPower + thermal_units[i].DeltaRampUp)*is_on[i, t].out - (thermal_units[i].MinPower)*is_on[i, t-1].out
                [i = 1: N], power[i].in - power[i].out <= (-thermal_units[i].DeltaRampDown)*start_down[i,t].out + (thermal_units[i].MinPower + thermal_units[i].DeltaRampDown)*is_on[i, t-1].out - (thermal_units[i].MinPower)*is_on[i, t].out
                [line in Lines], flow[line.b1,line.b2] <= line.Fmax
                [line in Lines], flow[line.b1,line.b2] >= -line.Fmax
                [line in Lines], flow[line.b1,line.b2] == line.B12*(θ[line.b1]-θ[line.b2])
            end
        )
        cstr = [@constraint(subproblem,sum(power[unit.name].out for unit in thermal_units if unit.Bus==b) - power_curtailement[b] + power_shedding[b] == instance.Demandbus[b][t]*(1+force*sum(error_forecast[w] for w in 1:NumWindfarms if BusWind[w]==b))+sum(flow[b,bp] for bp in Next[b])) for b in Buses]
        # Stage-objective
        @stageobjective(subproblem, sum(unit.LinearTerm*power[unit.name].out for unit in thermal_units)+sum(MSUC.SHEDDING_COST*power_shedding[b]+MSUC.CURTAILEMENT_COST*power_curtailement[b] for b in Buses))

        end
    return cstr, z_var, gamma_var
end