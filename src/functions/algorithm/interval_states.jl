# --- Etat par intervalle de la passe avant : point de linearisation ytilde^k (Sect. 4.2) --------------
#
# Les coupes intervalle linearisent le probleme local de chaque intervalle [a:b] en un point
# ytilde^k_{j,a,b,t} : tout point atteignable donne une coupe valide, serree au point visite pour les
# intervalles choisis. Pour les autres, ce point n'influe que sur la qualite de la coupe aux plannings ou
# ils seraient choisis. Il est construit pendant la passe avant. Les intervalles choisis gardent la
# production echantillonnee ; pour les autres, `train(...; interval_rule)` choisit la regle, parmi
# `INTERVAL_RULES`, qui prend un point de l'ensemble des successeurs [lo, hi] de ytilde_{s-1} :
#
#   :dual        regle des prix (eq. (55)-(56)), la regle par defaut : chaque intervalle suit la trajectoire qui
#                minimise son cout lineaire local, c'est-a-dire monte a fond (hi) si son signal de cout z est
#                negatif et descend a fond (lo) sinon. Un intervalle indifferent (|z| <= SIGNAL_TOLERANCE,
#                typiquement l'unite marginale, dont la production est fixee par l'equilibre offre-demande et non
#                par son cout) prend la production echantillonnee de son unite, ramenee dans [lo, hi] ;
#   :projection  la production echantillonnee de l'unite, projetee sur [lo, hi] ;
#   :minimum     toujours le plus petit niveau atteignable, lo ;
#   :maximum     toujours le plus grand niveau atteignable, hi.
#
# Seule la regle :dual lit les multiplicateurs du probleme d'etape (donc seule elle calcule les duaux).

# Un signal de cout dont la valeur absolue est au plus SIGNAL_TOLERANCE est considere nul : les duaux ne sont
# connus qu'a la tolerance d'optimalite du solveur (1e-6 pour Gurobi), donc son signe n'a pas de sens.
const SIGNAL_TOLERANCE = 1e-6

# Les regles de construction de ytilde^k pour les intervalles non choisis (`train(...; interval_rule)`).
const INTERVAL_RULES = (:dual, :projection, :minimum, :maximum)

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
    forward_interval_state(model, node, previous, y, duals, rule = :dual) -> (state, signal)

Etat par intervalle ytilde^k_{.,s} (`state`) a la sortie de l'etape `s = node.index - 1` de la passe avant, a
partir de celui de l'etape precedente `previous`, de la production echantillonnee `y` (une par unite) et des
multiplicateurs `duals` du probleme d'etape resolu (`nothing` si `rule != :dual`, qui n'en a pas besoin).
`signal` est le signal de cout z^F_{j,a,b,s} de chaque intervalle actif a l'etape s (NaN pour les autres, et
partout si `rule != :dual`), calcule aussi pour les intervalles choisis. Pour chaque intervalle [a:b] actif a
l'etape s :

  * choisi par le planning courant : la production echantillonnee, ytilde_s = y_{j,s} ;
  * sinon, un point de l'ensemble des successeurs (50) de ytilde_{s-1}, de bornes
    hi = min{Pbar_{a,b,s}, ytilde_{s-1} + R^U_{a,b,s}} et lo = max{P_j, ytilde_{s-1} - R^D}, donne par `rule`
    (`INTERVAL_RULES`) :
      - `:dual` : avec le signal de cout (55)

            z_{j,a,b,s} = c_{j,s} - nu_{n(j),s} + sum_l sigma_l betatilde^l_{j,a,b,s+1}

        hi si z < -SIGNAL_TOLERANCE (eq. (56)), lo si z > SIGNAL_TOLERANCE ; si |z| <= SIGNAL_TOLERANCE,
        l'intervalle est indifferent : la production echantillonnee y_{j,s} de l'unite, projetee sur [lo, hi]
        (le successeur le plus proche de ce que le dispatch demande) ;
      - `:projection` : la production echantillonnee y_{j,s} de l'unite projetee sur [lo, hi] ;
      - `:minimum` : lo ;
      - `:maximum` : hi.

Un intervalle inactif a l'etape s vaut 0 (Hypothese 1) et n'a pas de signal.
"""
function forward_interval_state(
    model::PolicyGraph,
    node::Node,
    previous,
    y,
    duals::Union{StageDuals,Nothing},
    rule::Symbol = :dual,
)
    rule in INTERVAL_RULES ||
        throw(ArgumentError("`interval_rule` must be one of $(INTERVAL_RULES). Got :$rule."))
    rule == :dual && duals === nothing &&
        throw(ArgumentError("the :dual rule needs the duals of the stage problem."))
    instance = model.instance
    idx = instance.intervals
    units = instance.Thermalunits
    s = node.index - 1
    horizon = instance.TimeHorizon
    selected = BitSet(model.schedule.selected)
    σβ = rule == :dual ? _weighted_sum(node.cuts, duals.σ, :beta) : nothing   # sum_l sigma_l betatilde^l_{.,s+1}, par intervalle
    state = zeros(length(previous))
    signal = fill(NaN, length(previous))
    for id in eachindex(state)
        a, b = idx.a[id], idx.b[id]
        is_on_interval(a, b, s) || continue
        unit = units[idx.unit[id]]
        if rule == :dual
            signal[id] = unit.LinearTerm - duals.ν[unit.Bus] + σβ[id]
        end
        if id in selected
            state[id] = y[unit.name]
            continue
        end
        lo = max(p_lower(unit, a, b, s), previous[id] - unit.DeltaRampDown)
        hi = min(p_upper(unit, a, b, s, horizon), previous[id] + ramp_up(unit, a, b, s))
        projected = min(max(y[unit.name], lo), hi)
        z = signal[id]
        state[id] =
            rule == :minimum ? lo :
            rule == :maximum ? hi :
            rule == :projection ? projected :
            z < -SIGNAL_TOLERANCE ? hi : z > SIGNAL_TOLERANCE ? lo : projected
    end
    return state, signal
end
