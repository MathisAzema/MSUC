# --- Etat par intervalle de la passe avant : regle des prix (Sect. 4.2, eq. (55)-(56)) ------------
#
# Les coupes intervalle linearisent le probleme local de chaque intervalle [a:b] en un point
# ytilde^k_{j,a,b,t} : tout point atteignable donne une coupe valide, serree au point visite pour les
# intervalles choisis. Pour les autres, ce point n'influe que sur la qualite de la coupe aux plannings ou
# ils seraient choisis. Il est construit pendant la passe avant, a partir des multiplicateurs du probleme
# d'etape echantillonne : chaque intervalle suit la trajectoire qui minimise son cout lineaire local,
# c'est-a-dire monte a fond si son signal de cout est negatif et descend a fond sinon. Les intervalles
# choisis gardent la production echantillonnee. Un intervalle indifferent (|z| <= SIGNAL_TOLERANCE, typiquement
# l'unite marginale, dont la production est fixee par l'equilibre offre-demande et non par son cout) prend la
# production echantillonnee de son unite, ramenee dans ce que l'intervalle peut atteindre.

# Un signal de cout dont la valeur absolue est au plus SIGNAL_TOLERANCE est considere nul : les duaux ne sont
# connus qu'a la tolerance d'optimalite du solveur (1e-6 pour Gurobi), donc son signe n'a pas de sens.
const SIGNAL_TOLERANCE = 1e-6

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
    forward_interval_state(model, node, previous, y, duals) -> (state, signal)

Etat par intervalle ytilde^k_{.,s} (`state`) a la sortie de l'etape `s = node.index - 1` de la passe avant, a
partir de celui de l'etape precedente `previous`, de la production echantillonnee `y` (une par unite) et des
multiplicateurs `duals` du probleme d'etape resolu. `signal` est le signal de cout z^F_{j,a,b,s} de chaque
intervalle actif a l'etape s (NaN pour les autres), calcule aussi pour les intervalles choisis. Pour chaque
intervalle [a:b] actif a l'etape s :

  * choisi par le planning courant : la production echantillonnee, ytilde_s = y_{j,s} ;
  * sinon : le signal de cout (55)

        z_{j,a,b,s} = c_{j,s} - nu_{n(j),s} + sum_l sigma_l betatilde^l_{j,a,b,s+1}

    donne la sortie de cout minimal dans l'ensemble des successeurs (50), de bornes
    hi = min{Pbar_{a,b,s}, ytilde_{s-1} + R^U_{a,b,s}} et lo = max{P_j, ytilde_{s-1} - R^D} :
      - hi si z < -SIGNAL_TOLERANCE (eq. (56)) ;
      - lo si z > SIGNAL_TOLERANCE ;
      - si |z| <= SIGNAL_TOLERANCE, l'intervalle est indifferent : la production echantillonnee y_{j,s} de
        l'unite, projetee sur [lo, hi] (le successeur le plus proche de ce que le dispatch demande).

Un intervalle inactif a l'etape s vaut 0 (Hypothese 1) et n'a pas de signal.
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
    signal = fill(NaN, length(previous))
    for id in eachindex(state)
        a, b = idx.a[id], idx.b[id]
        is_on_interval(a, b, s) || continue
        unit = units[idx.unit[id]]
        z = unit.LinearTerm - duals.ν[unit.Bus] + σβ[id]
        signal[id] = z
        if id in selected
            state[id] = y[unit.name]
            continue
        end
        lo = max(p_lower(unit, a, b, s), previous[id] - unit.DeltaRampDown)
        hi = min(p_upper(unit, a, b, s, horizon), previous[id] + ramp_up(unit, a, b, s))
        state[id] =
            z < -SIGNAL_TOLERANCE ? hi :
            z > SIGNAL_TOLERANCE ? lo : min(max(y[unit.name], lo), hi)
    end
    return state, signal
end
