# --- Etat par intervalle de la passe avant : regle des prix (Sect. 4.2, eq. (55)-(56)) ------------
#
# Les coupes intervalle linearisent le probleme local de chaque intervalle [a:b] en un point
# ytilde^k_{j,a,b,t} : tout point atteignable donne une coupe valide, serree au point visite pour les
# intervalles choisis. Pour les autres, ce point n'influe que sur la qualite de la coupe aux plannings ou
# ils seraient choisis. Avec `train(...; linearization = :price)`, il est construit pendant la passe avant,
# a partir des multiplicateurs du probleme d'etape echantillonne : chaque intervalle suit la trajectoire
# qui minimise son cout lineaire local, c'est-a-dire monte a fond si son signal de cout est negatif et
# descend a fond sinon. Les intervalles choisis gardent la production echantillonnee. La regle par
# defaut (`:max_reachable`, `linearization_point`) est calculee a la passe arriere, sans prix.

"""
    uses_price_rule(model)

Vrai si le modele construit l'etat par intervalle de la passe avant : formulation intervalle et
`train(...; linearization = :price)`.
"""
uses_price_rule(model::PolicyGraph) =
    model.method == ExtendedFormulation && get(model.ext, :linearization, :max_reachable) == :price

"""
    initial_interval_state(instance) -> Vector{Float64}

ytilde_{j,a,b,0} de l'eq. (17) pour tous les intervalles (identifiants de `IntervalIndex`) : la production
initiale du groupe si l'intervalle couvre t = 0, zero sinon.
"""
function initial_interval_state(instance::Instance)
    idx = instance.intervals
    return [
        initial_interval_output(instance.Thermalunits[idx.unit[id]], idx.a[id]) for
        id in eachindex(idx.unit)
    ]
end

"""
    forward_interval_state(model, node, previous, y, duals) -> Vector{Float64}

Etat par intervalle ytilde^k_{.,s} a la sortie de l'etape `s = node.index - 1` de la passe avant, a partir
de celui de l'etape precedente `previous`, de la production echantillonnee `y` (une par unite) et des
multiplicateurs `duals` du probleme d'etape resolu. Pour chaque intervalle [a:b] actif a l'etape s :

  * choisi par le planning courant : la production echantillonnee, ytilde_s = y_{j,s} ;
  * sinon : le signal de cout (55)

        z_{j,a,b,s} = c_{j,s} - nu_{n(j),s} + sum_l sigma_l betatilde^l_{j,a,b,s+1}

    donne la sortie de cout minimal dans l'ensemble des successeurs (50) : la borne haute
    min{Pbar_{a,b,s}, ytilde_{s-1} + R^U_{a,b,s}} si z < 0, la borne basse max{P_j, ytilde_{s-1} - R^D}
    sinon (eq. (56)).

Un intervalle inactif a l'etape s vaut 0 (Hypothese 1).
"""
function forward_interval_state(model::PolicyGraph, node::Node, previous, y, duals::StageDuals)
    instance = model.instance
    idx = instance.intervals
    units = instance.Thermalunits
    s = node.index - 1
    horizon = instance.TimeHorizon
    selected = BitSet(model.schedule.selected)
    σβ = _weighted_sum(node.cuts, duals.σ, :beta)      # sum_l sigma_l betatilde^l_{.,s+1}, par intervalle
    state = zeros(length(previous))
    for id in eachindex(state)
        a, b = idx.a[id], idx.b[id]
        is_on_interval(a, b, s) || continue
        unit = units[idx.unit[id]]
        if id in selected
            state[id] = y[unit.name]
            continue
        end
        z = unit.LinearTerm - duals.ν[unit.Bus] + σβ[id]
        state[id] =
            z < 0 ? min(p_upper(unit, a, b, s, horizon), previous[id] + ramp_up(unit, a, b, s)) :
            max(p_lower(unit, a, b, s), previous[id] - unit.DeltaRampDown)
    end
    return state
end
