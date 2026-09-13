# =============================================================================
#  Sous-problemes d'etape des deux formulations.
#
#  Noeud 1 (racine)  : le probleme d'engagement, (7) en 3-bin et (26) en intervalle.
#                      C'est le seul noeud non convexe.
#  Noeuds t+1 >= 2   : le probleme de dispatch de l'etape t, (2) resp. (41).
#
#  Les variables d'etat portees d'un noeud a l'autre sont :
#    x_j = (u_j, v_j, w_j)  planning 3-bin, fixe des la racine   (Sect. 2.1)
#    y_{j,t}                production physique de l'unite j     (Sect. 2.1)
#  et, pour la formulation intervalle, gamma_{j,a,b} (Def. 1) reconstruit dans
#  chaque sous-probleme a partir de x par les contraintes de liaison (26c).
# =============================================================================

"Taille du support de l'alea xi_t (Hypothese 1)."
const NOISE_SUPPORT_SIZE = 11

"Demi-amplitude de l'erreur relative de prevision de demande nette."
const NOISE_SPREAD = 0.05

"""
    forecast_error_support(instance; S, spread) -> (Ω, P)

Support fini et probabilites de l'alea xi_t (Hypothese 1) : erreur relative de
prevision, de meme loi a chaque etape et independante entre etapes.

Les DEUX formulations passent par cette fonction, sans quoi elles ne decriraient pas
le meme probleme et la Prop. 3 ne serait pas testable.
"""
function forecast_error_support(
    instance;
    S::Int = NOISE_SUPPORT_SIZE,
    spread::Float64 = NOISE_SPREAD,
)
    nw = length(instance.BusWind)
    S <= 1 && return ([zeros(nw)], [1.0])
    Ω = [[-spread + 2 * spread * (s - 1) / (S - 1) for _ in 1:nw] for s in 1:S]
    return Ω, fill(1 / S, S)
end

# Variables d'etat communes aux deux formulations.
function _declare_states!(subproblem, instance)
    N, T = instance.N, instance.TimeHorizon
    units = instance.Thermalunits
    @variable(
        subproblem,
        0.0 <= power[i = 1:N] <= units[i].MaxPower,
        MSUC.State,
        initial_value = units[i].InitialPower
    )
    @variable(subproblem, 0.0 <= is_on[i = 1:N, t in 0:T] <= 1.0, MSUC.State, initial_value = 0.0)
    @variable(subproblem, 0.0 <= start_up[i = 1:N, t in 1:T] <= 1.0, MSUC.State, initial_value = 0.0)
    @variable(subproblem, 0.0 <= start_down[i = 1:N, t in 1:T] <= 1.0, MSUC.State, initial_value = 0.0)
    return power, is_on, start_up, start_down
end

# Contraintes d'engagement de la racine : (1a), duree minimale de marche et d'arret,
# conditions initiales. Le cout d'engagement est l'objectif d'etape (1i).
function _commitment_constraints!(subproblem, instance, is_on, start_up, start_down)
    T = instance.TimeHorizon
    units = instance.Thermalunits
    for i in 1:instance.N
        JuMP.set_binary(is_on[i, 0].out)
        for t in 1:T
            JuMP.set_binary(is_on[i, t].out)
            JuMP.set_binary(start_up[i, t].out)
            JuMP.set_binary(start_down[i, t].out)
        end
    end
    # (1a) : u_{j,t} - u_{j,t-1} = v_{j,t} - w_{j,t}
    @constraint(
        subproblem,
        [unit in units, t in 1:T],
        is_on[unit.name, t].out - is_on[unit.name, t-1].out ==
        start_up[unit.name, t].out - start_down[unit.name, t].out
    )
    @constraint(subproblem, [unit in units, t in 1:T], start_up[unit.name, t].out <= is_on[unit.name, t].out)
    @constraint(subproblem, [unit in units, t in 1:T], start_down[unit.name, t].out <= 1 - is_on[unit.name, t].out)
    # (1b) : duree minimale de marche, et sa symetrique pour l'arret. Le terme
    # constant impose la fin de la duree minimale heritee des conditions initiales.
    @constraint(
        subproblem,
        [unit in units, t in 1:T; unit.InitUpDownTime !== nothing],
        sum(start_up[unit.name, τ].out for τ in max(1, t - unit.MinUpTime + 1):t) +
        1 * (t < unit.MinUpTime - unit.InitUpDownTime + 1) * (unit.InitUpDownTime > 0) <=
        is_on[unit.name, t].out
    )
    @constraint(
        subproblem,
        [unit in units, t in 1:T; unit.InitUpDownTime !== nothing],
        sum(start_down[unit.name, τ].out for τ in max(1, t - unit.MinDownTime + 1):t) +
        1 * (t < unit.MinDownTime + unit.InitUpDownTime + 1) * (unit.InitUpDownTime < 0) <=
        1 - is_on[unit.name, t].out
    )
    @constraint(
        subproblem,
        [unit in units, t in 1:T; unit.InitUpDownTime === nothing],
        sum(start_up[unit.name, τ].out for τ in max(1, t - unit.MinUpTime + 1):t) <=
        is_on[unit.name, t].out
    )
    @constraint(
        subproblem,
        [unit in units, t in 1:T; unit.InitUpDownTime === nothing],
        sum(start_down[unit.name, τ].out for τ in max(1, t - unit.MinDownTime + 1):t) <=
        1 - is_on[unit.name, t].out
    )
    @constraint(
        subproblem,
        [unit in units; unit.InitUpDownTime !== nothing],
        is_on[unit.name, 0].out == (unit.InitUpDownTime >= 0)
    )
    @constraint(
        subproblem,
        [unit in units; unit.InitUpDownTime === nothing],
        is_on[unit.name, 0].out == 0
    )
    # Une unite demarree au-dessus de MinPower ne peut pas s'arreter avant d'avoir
    # eu le temps de redescendre : elle reste allumee jusqu'a `limit`.
    for unit in units
        limit =
            (unit.InitialPower - unit.MinPower) / unit.DeltaRampDown < 0 ? -1 :
            Int64(ceil((unit.InitialPower - unit.MinPower) / unit.DeltaRampDown))
        for t in 0:limit
            @constraint(subproblem, is_on[unit.name, t].out == 1)
        end
    end
    @stageobjective(
        subproblem,
        sum(
            unit.ConstTerm * is_on[unit.name, t].out +
            unit.StartUpCost * start_up[unit.name, t].out +
            unit.StartDownCost * start_down[unit.name, t].out for unit in units for t in 1:T
        )
    )
    return
end

# Reseau, alea et bilan de demande d'une etape de dispatch : (1e)-(1h).
# Renvoie les contraintes de bilan, dont les duaux sont le nu du papier.
function _dispatch_common!(subproblem, instance, force, power, t)
    N, T = instance.N, instance.TimeHorizon
    units = instance.Thermalunits
    Next = instance.Next
    Buses = 1:size(Next)[1]
    Lines = values(instance.Lines)
    BusWind = instance.BusWind
    nw = length(BusWind)

    @variables(subproblem, begin
        power_shedding[b in Buses] >= 0
        power_curtailement[b in Buses] >= 0
        θ[b in Buses]
        flow[b in Buses, bp in Next[b]]
    end)
    # Alea : erreur relative de prevision, Hypothese 1.
    @variable(subproblem, error_forecast[k in 1:nw])
    Ω, P = forecast_error_support(instance)
    MSUC.parameterize(subproblem, Ω, P) do ω
        for k in 1:nw
            JuMP.fix(error_forecast[k], ω[k])
        end
    end
    # (1e)-(1f) : flux DC et capacites de ligne.
    @constraints(subproblem, begin
        [line in Lines], flow[line.b1, line.b2] <= line.Fmax
        [line in Lines], flow[line.b1, line.b2] >= -line.Fmax
        [line in Lines], flow[line.b1, line.b2] == line.B12 * (θ[line.b1] - θ[line.b2])
    end)
    # (1h) : bilan par noeud. Les duaux nu_{n,t} alimentent z_{j,a,b} (Prop. 4).
    demand = [
        @constraint(
            subproblem,
            sum(power[unit.name].out for unit in units if unit.Bus == b) -
            power_curtailement[b] + power_shedding[b] ==
            instance.Demandbus[b][t] *
            (1 + force * sum(error_forecast[w] for w in 1:nw if BusWind[w] == b)) +
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

# Capacite et rampes sur la production physique : (43a)-(43c) ecrites sur x_j.
function _capacity_and_ramping!(subproblem, instance, power, is_on, start_up, start_down, t)
    units = instance.Thermalunits
    @constraints(subproblem, begin
        # (43a)
        [i = 1:instance.N], power[i].out <= units[i].MaxPower * is_on[i, t].out
        [i = 1:instance.N], power[i].out >= units[i].MinPower * is_on[i, t].out
        # (43b) : au demarrage S^U = MinPower, en marche normale R^U
        [i = 1:instance.N],
        power[i].out - power[i].in <=
        (-units[i].DeltaRampUp) * start_up[i, t].out +
        (units[i].MinPower + units[i].DeltaRampUp) * is_on[i, t].out -
        units[i].MinPower * is_on[i, t-1].out
        # (43c) : a l'arret S^D = MinPower, en marche normale R^D
        [i = 1:instance.N],
        power[i].in - power[i].out <=
        (-units[i].DeltaRampDown) * start_down[i, t].out +
        (units[i].MinPower + units[i].DeltaRampDown) * is_on[i, t-1].out -
        units[i].MinPower * is_on[i, t].out
    end)
    return
end

"""
    subproblem_builder_UC_def(instance, force, subproblem, node)

Sous-probleme de la formulation 3-bin de reference (Sect. 2). La faisabilite d'arret
est imposee par les lignes (44), une par decalage h : le planning 3-bin ne porte pas
l'instant d'arret, il faut donc enumerer les decalages.
"""
function subproblem_builder_UC_def(instance::MSUC.Instance, force::Float64, subproblem::Model, node::Int)
    N, T = instance.N, instance.TimeHorizon
    units = instance.Thermalunits
    power, is_on, start_up, start_down = _declare_states!(subproblem, instance)

    if node == 1
        _commitment_constraints!(subproblem, instance, is_on, start_up, start_down)
        @constraint(subproblem, [i in 1:N], power[i].out == power[i].in)
        return JuMP.ConstraintRef[]
    end

    t = node - 1
    @constraint(subproblem, [i in 1:N, k in 0:T], is_on[i, k].out == is_on[i, k].in)
    @constraint(subproblem, [i in 1:N, k in 1:T], start_up[i, k].out == start_up[i, k].in)
    @constraint(subproblem, [i in 1:N, k in 1:T], start_down[i, k].out == start_down[i, k].in)

    # (44) : faisabilite d'arret. Seuls les decalages h dont le coefficient est
    # strictement positif portent de l'information ; au-dela la ligne se reduit a (43a).
    for unit in units
        for h in 1:(T-t)
            coef = max(0.0, unit.MaxPower - unit.MinPower - (h - 1) * unit.DeltaRampDown)
            # Au-dela, le coefficient s'annule et la ligne se reduit a (43a).
            coef <= 0 && break
            @constraint(
                subproblem,
                power[unit.name].out <=
                unit.MaxPower * is_on[unit.name, t].out +
                coef * (1 - is_on[unit.name, t].out - start_down[unit.name, t+h].out)
            )
        end
    end
    _capacity_and_ramping!(subproblem, instance, power, is_on, start_up, start_down, t)
    return _dispatch_common!(subproblem, instance, force, power, t)
end

"""
    subproblem_builder_UC_extended(instance, force, subproblem, node)

Sous-probleme de la formulation intervalle (Sect. 3). Renvoie `(bilan, gamma)`.

Les variables gamma_{j,a,b} (Def. 1) sont reconstruites dans chaque sous-probleme a
partir du planning 3-bin par les liaisons (26c) ; seule la racine les optimise, les
autres noeuds les heritent via l'etat x. La faisabilite d'arret est la ligne (46),
une par intervalle : l'instant d'arret y est explicite, aucun decalage a enumerer.

L'etat de production reste y_{j,t} (une variable par unite) et non ytilde_{j,a,b,t} :
l'identite ytilde = u_{a,b,t} y_{j,t} permet d'ecrire (41g) au planning courant, ce
que fait la passe avant en reinstanciant les coupes.
"""
function subproblem_builder_UC_extended(instance::MSUC.Instance, force::Float64, subproblem::Model, node::Int)
    N, T = instance.N, instance.TimeHorizon
    units = instance.Thermalunits
    power, is_on, start_up, start_down = _declare_states!(subproblem, instance)

    # gamma_{j,a,b} >= 0, [a:b] in T_j.
    gamma = Dict{Tuple{Int,Int,Int},JuMP.VariableRef}()
    for i in 1:N, (a, b) in units[i].intervals
        gamma[i, a, b] = @variable(subproblem, lower_bound = 0.0, base_name = "gamma_$(i)_$(a)_$(b)")
    end
    # (26c) : x_j = sum_{[a:b]} gamma_{j,a,b} x_{a,b}, composante par composante (15).
    @constraint(subproblem, [unit in units, t in 1:T],
        sum(gamma[unit.name, a, b] for (a, b) in unit.intervals if a <= t < b) == is_on[unit.name, t].out)
    @constraint(subproblem, [unit in units, t in 1:T],
        sum(gamma[unit.name, a, b] for (a, b) in unit.intervals if a == t) == start_up[unit.name, t].out)
    @constraint(subproblem, [unit in units, t in 1:T],
        sum(gamma[unit.name, a, b] for (a, b) in unit.intervals if b == t) == start_down[unit.name, t].out)

    if node == 1
        _commitment_constraints!(subproblem, instance, is_on, start_up, start_down)
        @constraint(subproblem, [i in 1:N], power[i].out == power[i].in)
        return JuMP.ConstraintRef[], gamma
    end

    t = node - 1
    @constraint(subproblem, [i in 1:N, k in 0:T], is_on[i, k].out == is_on[i, k].in)
    @constraint(subproblem, [i in 1:N, k in 1:T], start_up[i, k].out == start_up[i, k].in)
    @constraint(subproblem, [i in 1:N, k in 1:T], start_down[i, k].out == start_down[i, k].in)

    # (46) agregee par l'identite ytilde = u_{a,b,t} y_{j,t} : la borne d'arret de
    # chaque intervalle actif, ponderee par gamma.
    @constraint(subproblem, [i = 1:N],
        power[i].out <=
        sum(min(units[i].MaxPower, units[i].MinPower + units[i].DeltaRampDown * (b - t - 1)) * gamma[i, a, b]
            for (a, b) in units[i].intervals_t[t] if b <= T) +
        sum(units[i].MaxPower * gamma[i, a, b] for (a, b) in units[i].intervals_t[t] if b > T))

    _capacity_and_ramping!(subproblem, instance, power, is_on, start_up, start_down, t)
    return _dispatch_common!(subproblem, instance, force, power, t), gamma
end
