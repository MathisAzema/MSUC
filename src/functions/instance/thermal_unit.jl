# Standard capacity + ramp model (Sect. 4) for unit j, interval [a:b] in T_j and stage t.
# Paper symbol -> function: u_{a,b,t} (15a) -> is_on_interval, R^U (45b) -> ramp_up, R^D (45c) -> ramp_down,
# q (49) -> reference_output, ytilde_{j,a,b,0} (17) -> initial_interval_output, Qhat (48)-(49) -> local_support.
# Here S^U_j = S^D_j = MinPower.

is_on_interval(a, b, t) = a <= t <= b - 1

"""
    set_intervals(MinUpTime, InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)

Admissible operating intervals T_j of a unit (Sect. 3.1): all `[a:b]` with `1 <= a < b <= T+1`
respecting the minimum up time, plus the initial interval if the unit starts on.
`b = T+1` is always admitted, even below the minimum up time (the unit stays on past the horizon).
"""
function set_intervals(MinUpTime, InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)
    bmax = T + 1
    intervals = Vector{Int64}[]
    for a in 1:T
        for b in min(a + MinUpTime, bmax):bmax
            push!(intervals, [a, b])
        end
    end
    if InitUpDownTime >= 0
        a0 = -InitUpDownTime + 1
        bmin = max(
            1,
            a0 + MinUpTime,
            Int64(1 + ceil((InitialPower - MinPower) / DeltaRampDown)),
        )
        for b in min(bmin, bmax):bmax
            push!(intervals, [a0, b])
        end
    end
    return intervals
end

p_lower(unit, a, b, t) = is_on_interval(a, b, t) ? unit.MinPower : 0.0

# Local upper bound (Sect. 4.2), including the shutdown line (46). The startup cone is not
# included here: it belongs to the reference trajectory, see `reference_output`.
function p_upper(unit, a, b, t, T)
    is_on_interval(a, b, t) || return 0.0
    return b <= T ?
           min(unit.MaxPower, unit.MinPower + (b - t - 1) * unit.DeltaRampDown) :
           unit.MaxPower
end

# R^U_{j,a,b,t}: right-hand side of (45b) on x_{a,b}.
function ramp_up(unit, a, b, t)
    if t == a
        return unit.MinPower
    elseif t < a || t > b
        return 0.0
    elseif t == b
        return -unit.MinPower
    else
        return unit.DeltaRampUp
    end
end

# R^D_{j,a,b,t}: right-hand side of (45c) on x_{a,b}.
function ramp_down(unit, a, b, t)
    if t == b
        return unit.MinPower
    elseif t < a || t > b
        return 0.0
    elseif t == a
        return -unit.MinPower
    else
        return unit.DeltaRampDown
    end
end

initial_interval_output(unit, a) = a <= 0 ? unit.InitialPower : 0.0

# q_{j,a,b,t} (eq. (52)): the largest output the interval admits, i.e. the trajectory of an interval
# whose cost signal is negative at every stage. It gives the coefficients of the first cuts, when all
# balance prices equal the load-shedding cost.
function reference_output(unit, a, b, t, T)
    t == 0 && return initial_interval_output(unit, a)
    is_on_interval(a, b, t) || return 0.0
    startup = a >= 1 ? unit.MinPower + (t - a) * unit.DeltaRampUp :
                       unit.InitialPower + t * unit.DeltaRampUp
    return b <= T ?
           min(startup, unit.MaxPower, unit.MinPower + (b - t - 1) * unit.DeltaRampDown) :
           min(startup, unit.MaxPower)
end

"""
    local_support(unit, a, b, t, T, y_prev, z) -> (h, beta)

Closed-form affine support (47)-(49) of the local problem Qhat_{j,a,b,t}(., z), with
`Qhat(y_prev, z) == h + beta * y_prev`. `y_prev` is the linearization point: any value gives a
valid support, it only selects the branch (Prop. 4).
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
