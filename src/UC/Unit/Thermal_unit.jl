# =============================================================================
#  Section 4 du papier : le modele standard capacite + rampes.
#
#  Notations, pour l'unite j, l'intervalle [a:b] in T_j et l'etape t :
#
#    u_{a,b,t}           indicatrice "allumee", eq. (15a)          -> is_on_interval
#    p_lower, p_upper    bornes de capacite locales, Sect. 4.2     -> p_lower, p_upper
#    R^U_{j,a,b,t}       membre de droite de (45b)                 -> ramp_up
#    R^D_{j,a,b,t}       membre de droite de (45c)                 -> ramp_down
#    q_{j,a,b,t}         trajectoire de reference, eq. (49)        -> reference_output
#    ytilde_{j,a,b,0}    etat initial par intervalle, eq. (17)     -> initial_interval_output
#    Qhat_{j,a,b,t}      valeur locale (48) et support (49)        -> local_support
#
#  Dans ce modele S^U_j = S^D_j = MinPower : evaluer (45b)-(45c) sur x_{a,b} au
#  demarrage (t=a) et a l'arret (t=b) donne MinPower comme limite de rampe.
# =============================================================================

"""
    is_on_interval(a, b, t)

u_{a,b,t} de l'eq. (15a) : vaut 1 ssi t in [a : b-1].
"""
is_on_interval(a, b, t) = a <= t <= b - 1

"""
    set_intervals(MinUpTime, InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)

T_j : intervalles d'exploitation admissibles de l'unite (Sect. 3.1). Contient les
[a:b] avec 1 <= a < b <= T+1 respectant la duree minimale de marche, plus, si
l'unite est initialement allumee, l'intervalle initial [-a_0 : b].
"""
function set_intervals(MinUpTime, InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)
    bmax = T + 1
    intervals = [[a, b] for a in 1:T for b in a+MinUpTime:bmax]
    if InitUpDownTime !== nothing && InitUpDownTime >= 0
        bmin = max(
            1,
            -InitUpDownTime + 1 + MinUpTime,
            Int64(1 + ceil((InitialPower - MinPower) / DeltaRampDown)),
        )
        for b in bmin:bmax
            push!(intervals, [-InitUpDownTime + 1, b])
        end
    end
    return intervals
end

"""
    set_intervals_t(t, MinUpTime, InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)

Sous-ensemble de T_j actif a l'etape t, c.-a-d. les [a:b] avec a <= t <= b-1.
Utilise par la ligne d'arret (46) ecrite sur la puissance agregee.
"""
function set_intervals_t(t, MinUpTime, InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)
    bmax = T + 1
    intervals = [[a, b] for a in 1:t for b in max(t + 1, a + MinUpTime):bmax]
    if InitUpDownTime !== nothing && InitUpDownTime >= 0
        bmin = max(
            t + 1,
            -InitUpDownTime + 1 + MinUpTime,
            Int64(1 + ceil((InitialPower - MinPower) / DeltaRampDown)),
        )
        for b in bmin:bmax
            push!(intervals, [-InitUpDownTime + 1, b])
        end
    end
    return intervals
end

"""
    p_lower(unit, a, b, t)

Borne basse locale de (45a) : MinPower * u_{a,b,t}.
"""
p_lower(unit, a, b, t) = is_on_interval(a, b, t) ? unit.MinPower : 0.0

"""
    p_upper(unit, a, b, t, T)

Borne haute locale, Sect. 4.2 :
    min{Pmax, S^D + R^D (b-1-t)}  si b <= T,   Pmax  si b = T+1,
et 0 hors de l'intervalle. Le second terme est la ligne d'arret (46).

Le cone de demarrage n'entre PAS ici : il appartient a la trajectoire de
reference q, cf. `reference_output`.
"""
function p_upper(unit, a, b, t, T)
    is_on_interval(a, b, t) || return 0.0
    return b <= T ?
           min(unit.MaxPower, unit.MinPower + (b - t - 1) * unit.DeltaRampDown) :
           unit.MaxPower
end

"""
    ramp_up(unit, a, b, t)

R^U_{j,a,b,t} : membre de droite de (45b) evalue sur x_{a,b}. Vaut S^U au
demarrage (t=a), R^U en marche normale, -S^U a l'arret (t=b), 0 en dehors.
"""
function ramp_up(unit, a, b, t)
    if t == a
        return unit.MinPower          # S^U
    elseif t < a || t > b
        return 0.0
    elseif t == b
        return -unit.MinPower         # -S^U
    else
        return unit.DeltaRampUp       # R^U
    end
end

"""
    ramp_down(unit, a, b, t)

R^D_{j,a,b,t} : membre de droite de (45c) evalue sur x_{a,b}. Vaut S^D a l'arret
(t=b), R^D en marche normale, -S^D au demarrage (t=a), 0 en dehors.
"""
function ramp_down(unit, a, b, t)
    if t == b
        return unit.MinPower          # S^D
    elseif t < a || t > b
        return 0.0
    elseif t == a
        return -unit.MinPower         # -S^D
    else
        return unit.DeltaRampDown     # R^D
    end
end

"""
    initial_interval_output(unit, a)

ytilde_{j,a,b,0} de l'eq. (17) : la production initiale de l'unite si l'intervalle
couvre t=0 (a <= 0), zero sinon.
"""
initial_interval_output(unit, a) = a <= 0 ? unit.InitialPower : 0.0

"""
    reference_output(unit, a, b, t, T)

q_{j,a,b,t} de l'eq. (49) : trajectoire de reference de l'intervalle, c.-a-d. le
plus grand profil qu'il admet. Sert de point de linearisation pour les
intervalles NON selectionnes (Sect. 4.3). Nulle hors de [a : b-1] ; en t=0 elle
vaut l'etat initial (17).
"""
function reference_output(unit, a, b, t, T)
    t == 0 && return initial_interval_output(unit, a)
    is_on_interval(a, b, t) || return 0.0
    startup = a >= 1 ? unit.MinPower + (t - a) * unit.DeltaRampUp :   # S^U + R^U (t-a)
                       unit.InitialPower + t * unit.DeltaRampUp      # y_{j,0} + R^U t
    return b <= T ?
           min(startup, unit.MaxPower, unit.MinPower + (b - t - 1) * unit.DeltaRampDown) :
           min(startup, unit.MaxPower)
end

"""
    local_support(unit, a, b, t, T, y_prev, z) -> (h, beta)

Support affine du probleme local Qhat_{j,a,b,t}(., z), forme close (47)-(49).

Le domaine local est l'intervalle
    [ max(p_lower, y_prev - R^D) , min(p_upper, y_prev + R^U) ]   (47)
et le minimum de z*ytilde y est atteint a gauche si z >= 0, a droite sinon (48).
La branche affine qui realise ce maximum en `y_prev` minore Qhat partout ; on
renvoie sa constante h et sa pente beta (49), avec l'invariant

    Qhat(y_prev, z) == h + beta * y_prev.

`y_prev` est le point de LINEARISATION : n'importe quelle valeur donne un support
valide, elle ne fixe que la branche retenue (Prop. 4).
"""
function local_support(unit, a, b, t, T, y_prev::Float64, z::Float64)
    p_lo = p_lower(unit, a, b, t)
    p_hi = p_upper(unit, a, b, t, T)
    if z >= 0
        R_dn = ramp_down(unit, a, b, t)
        return y_prev - R_dn >= p_lo ? (-z * R_dn, z) : (z * p_lo, 0.0)
    else
        R_up = ramp_up(unit, a, b, t)
        return y_prev + R_up <= p_hi ? (z * R_up, z) : (z * p_hi, 0.0)
    end
end

function thermal_unit_commit_constraints(model, instance)
    """
    Define first-stage constraints in the 3-bin formulation
    """
    is_on=model[:is_on]
    start_up=model[:start_up]
    start_down=model[:start_down]
    thermal_fixed_cost=model[:thermal_fixed_cost]

    T= instance.TimeHorizon
    thermal_units=values(instance.Thermalunits)

    @constraint(model,  [unit in thermal_units, t in 1:T], is_on[unit.name, t]-is_on[unit.name, t-1]==start_up[unit.name, t]-start_down[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T], start_up[unit.name, t]<=is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T], start_down[unit.name, t]<=1-is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_up[unit.name, τ] for τ in max(1, t-unit.MinUpTime+1):t)+1*(t<unit.MinUpTime-unit.InitUpDownTime+1)*(unit.InitUpDownTime>0) <= is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_down[unit.name, τ] for τ in max(1, t-unit.MinDownTime+1):t)+1*(t<unit.MinDownTime+unit.InitUpDownTime+1)*(unit.InitUpDownTime<0) <= 1-is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_up[unit.name, τ] for τ in max(1, t-unit.MinUpTime+1):t) <= is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_down[unit.name, τ] for τ in max(1, t-unit.MinDownTime+1):t) <= 1-is_on[unit.name, t])

    @constraint(model,  [unit in thermal_units; unit.InitUpDownTime!=nothing], is_on[unit.name, 0]==(unit.InitUpDownTime>=0))

    @constraint(model,  [unit in thermal_units; unit.InitUpDownTime==nothing], is_on[unit.name, 0]==0)

    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T))

    for unit in thermal_units
        if (unit.InitialPower-unit.MinPower)/unit.DeltaRampDown <0
            limit=-1
        else
            limit=Int64(ceil((unit.InitialPower-unit.MinPower)/unit.DeltaRampDown))
        end
        for t in 0:limit
            @constraint(model, is_on[unit.name, t]==1)
        end
    end
end

function thermal_unit_commit_constraints_SDDP(subproblem, instance)
    """
    Define first-stage constraints in the 3-bin formulation
    """


    T= instance.TimeHorizon
    N=instance.N
    thermal_units=instance.Thermalunits

    is_on = subproblem[:is_on]
    start_up = subproblem[:start_up]
    start_down = subproblem[:start_down]

    gamma = subproblem[:gamma]

    @constraint(subproblem, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a<=t<b)==is_on[unit.name,t].out)
    @constraint(subproblem, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a==t)==start_up[unit.name,t].out)
    @constraint(subproblem, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if b==t)==start_down[unit.name,t].out)

    @constraint(subproblem,  [i in 1:N, t in 0:T], is_on[i, t].out == is_on[i, t].in)
    @constraint(subproblem,  [i in 1:N, t in 1:T], start_up[i, t].out == start_up[i, t].in)
    @constraint(subproblem,  [i in 1:N, t in 1:T], start_down[i, t].out == start_down[i, t].in)
end

function thermal_unit_commit_constraints_extended(model, instance)
    """
    Define first-stage constraints in the extended formulation
    """
    is_on=model[:is_on]
    start_up=model[:start_up]
    start_down=model[:start_down]
    thermal_fixed_cost=model[:thermal_fixed_cost]

    T= instance.TimeHorizon
    thermal_units=values(instance.Thermalunits)

    @constraint(model,  [unit in thermal_units, t in 1:T], is_on[unit.name, t]-is_on[unit.name, t-1]==start_up[unit.name, t]-start_down[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T], start_up[unit.name, t]<=is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T], start_down[unit.name, t]<=1-is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_down[unit.name, τ] for τ in max(1, t-unit.MinDownTime+1):t)+1*(t<unit.MinDownTime+unit.InitUpDownTime+1)*(unit.InitUpDownTime<0) <= 1-is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_down[unit.name, τ] for τ in max(1, t-unit.MinDownTime+1):t) <= 1-is_on[unit.name, t])

    @constraint(model,  [unit in thermal_units; unit.InitUpDownTime!=nothing], is_on[unit.name, 0]==(unit.InitUpDownTime>=0))

    @constraint(model,  [unit in thermal_units; unit.InitUpDownTime==nothing], is_on[unit.name, 0]==0)

    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T))

    for unit in thermal_units
        if (unit.InitialPower-unit.MinPower)/unit.DeltaRampDown <0
            limit=-1
        else
            limit=Int64(ceil((unit.InitialPower-unit.MinPower)/unit.DeltaRampDown))
        end
        for t in 0:limit
            @constraint(model, is_on[unit.name, t]==1)
        end
    end
end

function thermal_unit_capacity_constraints_scenarios(model, instance, S)
    """
    Define second-stage constraints in the extensive formulation
    """
    power=model[:power]
    is_on=model[:is_on]
    start_up=model[:start_up]
    start_down=model[:start_down]
    thermal_fuel_cost=model[:thermal_fuel_cost]
    power_shedding=model[:power_shedding]
    power_curtailement=model[:power_curtailement]

    T= instance.TimeHorizon
    Next=instance.Next
    Buses=1:size(Next)[1]
    thermal_units=values(instance.Thermalunits)

    @constraint(model,  [unit in thermal_units, t in 0:T, s in 1:S], power[unit.name, t, s]>=unit.MinPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 0:T, s in 1:S], power[unit.name, t, s]<=unit.MaxPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, s in 1:S; unit.InitialPower!=nothing], power[unit.name, 0, s]==unit.InitialPower)

    @constraint(model,  [unit in thermal_units, t in 1:T, s in 1:S], power[unit.name, t, s]-power[unit.name, t-1, s]<=(-unit.DeltaRampUp)*start_up[unit.name, t]+(unit.MinPower+unit.DeltaRampUp)*is_on[unit.name, t]-(unit.MinPower)*is_on[unit.name, t-1])
    @constraint(model,  [unit in thermal_units, t in 1:T, s in 1:S], power[unit.name, t-1, s]-power[unit.name, t, s]<=(-unit.DeltaRampDown)*start_down[unit.name, t]+(unit.MinPower+unit.DeltaRampDown)*is_on[unit.name, t-1]-(unit.MinPower)*is_on[unit.name, t])

    @constraint(model, [s in 1:S], thermal_fuel_cost[s]>=sum(unit.LinearTerm*power[unit.name, t, s] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t,s]+CURTAILEMENT_COST*power_curtailement[b,t,s] for b in Buses for t in 1:T))
end