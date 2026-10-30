# Capacite, rampes et faisabilite d'arret de l'etape t, ecrites sur le planning fixe.
function _stage_parameters!(subproblem, instance, power, t, basis)
    N, T = instance.N, instance.TimeHorizon
    units = instance.Thermalunits
    u_prev = @variable(subproblem, [1:N] in Parameter(0.0))
    u = @variable(subproblem, [1:N] in Parameter(0.0))
    v = @variable(subproblem, [1:N] in Parameter(0.0))
    w = @variable(subproblem, [1:N] in Parameter(0.0))
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
                p = @variable(subproblem, set = Parameter(0.0))
                push!(w_future[unit.name], p)
                @constraint(
                    subproblem,
                    power[unit.name].out <=
                    unit.MaxPower * u[unit.name] + coef * (1 - u[unit.name] - p)
                )
            end
        end
    else
        cap = @variable(subproblem, [i = 1:N] in Parameter(units[i].MaxPower))
        @constraint(subproblem, [i = 1:N], power[i].out <= cap[i])
    end
    return StageParameters(u_prev, u, v, w, w_future, cap)
end

# Reseau, alea et bilan de demande d'une etape de dispatch : (1c)-(1g).
# Renvoie les contraintes de bilan, dont les duaux sont le nu du papier.
function _dispatch_common!(subproblem, instance, force, power, ζ, t)
    units = instance.Thermalunits
    Next = instance.Next
    Buses = 1:size(Next)[1]
    arma = instance.arma
    nz = MSUC.zeta_dimension(arma)
    nb = length(Buses)

    @variables(subproblem, begin
        power_shedding[b in Buses] >= 0
        power_curtailement[b in Buses] >= 0
        θ[b in Buses]
        flow[b in Buses, bp in Next[b]]
    end)
    # Innovation xi_t a support fini (Hypothese 2) et recursion ARMA (1f) :
    #     zeta_t = Psi zeta_{t-1} + E xi_t
    @variable(subproblem, ξ[n = 1:nb])
    MSUC.parameterize(subproblem, arma.support, arma.probability) do ω
        for n in 1:nb
            JuMP.fix(ξ[n], ω[n])
        end
    end
    @constraint(
        subproblem,
        [i in 1:nz],
        ζ[i].out ==
        sum(arma.Psi[i, k] * ζ[k].in for k in 1:nz) +
        sum(arma.E[i, n] * ξ[n] for n in 1:nb)
    )
    # (1c)-(1e) : flux DC et capacites de ligne.
    for line in values(instance.Lines)
        add_line_constraints!(subproblem, line, flow[line.b1, line.b2], θ[line.b1], θ[line.b2])
    end
    # (1g) : bilan par noeud. Les duaux nu_{n,t} alimentent z_{j,a,b} (Prop. 4).
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
