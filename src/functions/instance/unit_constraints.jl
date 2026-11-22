# Constraints shared by the SDDP builder and the extensive formulations. They are written on
# arbitrary expressions (variables, parameters or states) so each model row is defined once.

"""
    add_commitment_constraints!(model, instance, u, v, w)

Commitment constraints on x = (u, v, w): transition (1a), minimum up/down times, initial
conditions. `u` is indexed `[unit.name, t]` on `0:T`; `v` and `w` on `1:T`.
"""
function add_commitment_constraints!(model, instance, u, v, w)
    T = instance.TimeHorizon
    units = instance.Thermalunits
    @constraint(model, [unit in units, t in 1:T], u[unit.name, t] - u[unit.name, t-1] == v[unit.name, t] - w[unit.name, t])
    @constraint(model, [unit in units, t in 1:T], v[unit.name, t] <= u[unit.name, t])
    @constraint(model, [unit in units, t in 1:T], w[unit.name, t] <= 1 - u[unit.name, t])
    @constraint(model, [unit in units, t in 1:T], sum(v[unit.name, τ] for τ in max(1, t-unit.MinUpTime+1):t) + 1 * (t < unit.MinUpTime - unit.InitUpDownTime + 1) * (unit.InitUpDownTime > 0) <= u[unit.name, t])
    @constraint(model, [unit in units, t in 1:T], sum(w[unit.name, τ] for τ in max(1, t-unit.MinDownTime+1):t) + 1 * (t < unit.MinDownTime + unit.InitUpDownTime + 1) * (unit.InitUpDownTime < 0) <= 1 - u[unit.name, t])
    @constraint(model, [unit in units], u[unit.name, 0] == (unit.InitUpDownTime >= 0))
    for unit in units
        limit = (unit.InitialPower - unit.MinPower) / unit.DeltaRampDown < 0 ? -1 :
                Int64(ceil((unit.InitialPower - unit.MinPower) / unit.DeltaRampDown))
        for t in 0:limit
            @constraint(model, u[unit.name, t] == 1)
        end
    end
    return
end

function commitment_cost(instance, u, v, w)
    return sum(
        unit.ConstTerm * u[unit.name, t] +
        unit.StartUpCost * v[unit.name, t] +
        unit.StartDownCost * w[unit.name, t] for unit in instance.Thermalunits for t in 1:instance.TimeHorizon
    )
end

# Capacity and ramp constraints (49a)-(49c) of one unit at stage t, with S^U = S^D = MinPower.
function add_unit_dynamics!(model, unit, y, y_prev, u, u_prev, v, w)
    @constraint(model, y <= unit.MaxPower * u)
    @constraint(model, y >= unit.MinPower * u)
    @constraint(model, y - y_prev <= -unit.DeltaRampUp * v + (unit.MinPower + unit.DeltaRampUp) * u - unit.MinPower * u_prev)
    @constraint(model, y_prev - y <= -unit.DeltaRampDown * w + (unit.MinPower + unit.DeltaRampDown) * u_prev - unit.MinPower * u)
    return
end

function add_line_constraints!(model, line, flow, θ_from, θ_to)
    @constraint(model, flow <= line.Fmax)
    @constraint(model, flow >= -line.Fmax)
    @constraint(model, flow == line.B12 * (θ_from - θ_to))
    return
end
