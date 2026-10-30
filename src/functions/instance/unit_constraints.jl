# =============================================================================
#  Contraintes partagees par le builder SDDP et les formulations extensives.
#
#  Elles sont ecrites sur des expressions quelconques (variables de decision,
#  parametres ou etats), pour que chaque ligne du modele n'existe qu'une fois :
#  `u`, `v`, `w` sont indexes `[unit.name, t]` ; `y`, `y_prev`, `u`, ... des
#  scalaires dans `add_unit_dynamics!`.
# =============================================================================

"""
    add_commitment_constraints!(model, instance, u, v, w)

Contraintes d'engagement sur x = (u, v, w) : equation de transition (1a), duree minimale
de marche et d'arret, conditions initiales, et maintien en marche d'une unite demarree au
dessus de MinPower le temps de redescendre. `u` est indexe sur `0:T`, `v` et `w` sur `1:T`.
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

"""
    commitment_cost(instance, u, v, w)

Cout d'engagement : cout a vide, de demarrage et d'arret sur tout l'horizon.
"""
function commitment_cost(instance, u, v, w)
    return sum(
        unit.ConstTerm * u[unit.name, t] +
        unit.StartUpCost * v[unit.name, t] +
        unit.StartDownCost * w[unit.name, t] for unit in instance.Thermalunits for t in 1:instance.TimeHorizon
    )
end

"""
    add_unit_dynamics!(model, unit, y, y_prev, u, u_prev, v, w)

Capacite et rampes de l'unite a l'etape t, ecrites sur le planning x = (u, v, w) :
production `y` a t, `y_prev` a t-1, `u` et `u_prev` le statut a t et t-1, `v` et `w` le
demarrage et l'arret a t. Avec S^U = S^D = MinPower, on retrouve (49a)-(49c).
"""
function add_unit_dynamics!(model, unit, y, y_prev, u, u_prev, v, w)
    @constraint(model, y <= unit.MaxPower * u)
    @constraint(model, y >= unit.MinPower * u)
    @constraint(model, y - y_prev <= -unit.DeltaRampUp * v + (unit.MinPower + unit.DeltaRampUp) * u - unit.MinPower * u_prev)
    @constraint(model, y_prev - y <= -unit.DeltaRampDown * w + (unit.MinPower + unit.DeltaRampDown) * u_prev - unit.MinPower * u)
    return
end

"""
    add_line_constraints!(model, line, flow, θ_from, θ_to)

Capacite de la ligne et flux DC : -Fmax <= flow <= Fmax et flow = B12 (θ_from - θ_to).
"""
function add_line_constraints!(model, line, flow, θ_from, θ_to)
    @constraint(model, flow <= line.Fmax)
    @constraint(model, flow >= -line.Fmax)
    @constraint(model, flow == line.B12 * (θ_from - θ_to))
    return
end
