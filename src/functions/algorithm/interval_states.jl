# Cost signals with |z| <= SIGNAL_TOLERANCE are treated as zero: duals are only known up to the
# solver's optimality tolerance (1e-6 for Gurobi), so their sign is meaningless.
const SIGNAL_TOLERANCE = 1e-6

# Rules to build ytilde^k for intervals not selected by the schedule (`train(...; interval_rule)`).
const INTERVAL_RULES = (:dual, :projection, :minimum, :maximum)

"""
    initial_interval_state(instance) -> Vector{Float64}

ytilde_{j,a,b,0} (eq. (17)) for all intervals: the unit's initial output if the interval covers t = 0,
zero otherwise.
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

Interval state ytilde^k_{.,s} at the exit of stage `s = node.index - 1`, from the previous one `previous`,
the sampled output `y` (one per unit) and the stage duals `duals` (`nothing` unless `rule == :dual`).
Also returns the cost signal z^F_{j,a,b,s} of each active interval (NaN otherwise, and everywhere unless
`rule == :dual`). For an interval [a:b] active at stage s:

  * selected by the schedule: the sampled output, ytilde_s = y_{j,s};
  * otherwise, a point of the successor set (50) of ytilde_{s-1}, with
    hi = min{Pbar_{a,b,s}, ytilde_{s-1} + R^U_{a,b,s}} and lo = max{P_j, ytilde_{s-1} - R^D}, given by `rule`:
      - `:dual`: with z_{j,a,b,s} = c_{j,s} - nu_{n(j),s} + sum_l sigma_l betatilde^l_{j,a,b,s+1} (eq. (55)),
        hi if z < -SIGNAL_TOLERANCE, lo if z > SIGNAL_TOLERANCE (eq. (56)), else (indifferent, typically
        the marginal unit) the sampled output projected on [lo, hi];
      - `:projection`: the sampled output projected on [lo, hi];
      - `:minimum`: lo;
      - `:maximum`: hi.

An interval inactive at stage s is 0 and has no signal.
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
    σβ = rule == :dual ? _weighted_sum(node.cuts, duals.σ, :beta) : nothing
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
