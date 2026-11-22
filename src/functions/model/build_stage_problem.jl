# Fixed variables carry the schedule x^k in the dispatch stages. They are not JuMP `Parameter`s:
# changing a `Parameter` makes JuMP copy the whole model into Gurobi at the next solve (losing the
# warm-start basis), whereas changing a fixed variable updates its bounds in place.
function _fixed_variables(subproblem, n, value)
    x = @variable(subproblem, [1:n])
    JuMP.fix.(x, value)
    return x
end

function _stage_parameters!(subproblem, instance, power, t, basis)
    N, T = instance.N, instance.TimeHorizon
    units = instance.Thermalunits
    u_prev = _fixed_variables(subproblem, N, 0.0)
    u = _fixed_variables(subproblem, N, 0.0)
    v = _fixed_variables(subproblem, N, 0.0)
    w = _fixed_variables(subproblem, N, 0.0)
    for i in 1:N
        add_unit_dynamics!(subproblem, units[i], power[i].out, power[i].in, u[i], u_prev[i], v[i], w[i])
    end

    w_future = [JuMP.VariableRef[] for _ in 1:N]
    cap = JuMP.VariableRef[]
    if basis === :bins
        for unit in units
            for h in 1:(T-t)
                coef = max(0.0, unit.MaxPower - unit.MinPower - (h - 1) * unit.DeltaRampDown)
                coef <= 0 && break
                p = _fixed_variables(subproblem, 1, 0.0)[1]
                push!(w_future[unit.name], p)
                @constraint(
                    subproblem,
                    power[unit.name].out <=
                    unit.MaxPower * u[unit.name] + coef * (1 - u[unit.name] - p)
                )
            end
        end
    else
        cap = _fixed_variables(subproblem, N, 0.0)
        JuMP.fix.(cap, [unit.MaxPower for unit in units])
        @constraint(subproblem, [i = 1:N], power[i].out <= cap[i])
    end
    return StageParameters(u_prev, u, v, w, w_future, cap)
end

# Network, noise and demand balance of a dispatch stage, (1c)-(1g). Returns the balance
# constraints, whose duals are the paper's nu.
function _dispatch_common!(subproblem, instance, force, power, ζ, t)
    units = instance.Thermalunits
    Next = instance.Next
    Buses = 1:size(Next)[1]
    arma = instance.arma
    nz = MSUC.zeta_dimension(arma)
    ni = size(arma.E, 2)

    @variables(subproblem, begin
        power_shedding[b in Buses] >= 0
        power_curtailement[b in Buses] >= 0
        θ[b in Buses]
        flow[b in Buses, bp in Next[b]]
    end)
    # Finite-support innovation xi_t and ARMA recursion (1f).
    @variable(subproblem, ξ[n = 1:ni])
    MSUC.parameterize(subproblem, arma.support, arma.probability) do ω
        for n in 1:ni
            JuMP.fix(ξ[n], ω[n])
        end
    end
    @constraint(
        subproblem,
        [i in 1:nz],
        ζ[i].out ==
        sum(arma.Psi[i, k] * ζ[k].in for k in 1:nz) +
        sum(arma.E[i, n] * ξ[n] for n in 1:ni)
    )
    for line in values(instance.Lines)
        add_line_constraints!(subproblem, line, flow[line.b1, line.b2], θ[line.b1], θ[line.b2])
    end
    # (1g) node balance; its duals nu_{n,t} feed z_{j,a,b} (Prop. 4).
    demand = [
        @constraint(
            subproblem,
            sum(power[unit.name].out for unit in units if unit.Bus == b) -
            power_curtailement[b] + power_shedding[b] ==
            instance.Demandbus[b][t] *
            (1 + force * sum(arma.S_eta[b, k] * ζ[k].out for k in 1:nz)) +
            sum(flow[b, bp] for bp in Next[b])
        ) for b in Buses
    ]
    @stageobjective(
        subproblem,
        sum(unit.LinearTerm * power[unit.name].out for unit in units) + sum(
            MSUC.SHEDDING_COST * power_shedding[b] +
            MSUC.CURTAILEMENT_COST * power_curtailement[b] for b in Buses
        )
    )
    return demand
end
