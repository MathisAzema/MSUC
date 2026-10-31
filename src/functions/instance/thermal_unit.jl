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
#    ytilde^k            point de linearisation, regle (55)        -> linearization_point
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

Convention de la Sect. 3.1 : **b = T+1 est toujours admis**, meme si la duree minimale
de marche n'est pas atteinte, pour representer une unite qui reste allumee au-dela de
l'horizon. Idem pour la contrainte de rampe d'arret de l'intervalle initial : il n'y a
pas d'arret en T+1.
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

"""
    set_intervals_t(t, MinUpTime, InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)

Sous-ensemble de T_j actif a l'etape t, c.-a-d. les [a:b] tels que u_{a,b,t} = 1.
Derive de `set_intervals` par filtrage, ce qui garantit l'inclusion T_j(t) subset T_j
dont depend l'indexation des variables gamma dans le builder.
"""
function set_intervals_t(t, MinUpTime, InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)
    all = set_intervals(MinUpTime, InitUpDownTime, InitialPower, MinPower, DeltaRampDown, T)
    return [ab for ab in all if is_on_interval(ab[1], ab[2], t)]
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
    linearization_point(unit, a, b, t, T, y)

Production ỹ^k de l'intervalle [a:b] a l'etape t ou l'on linearise le probleme local, regle (55)
de l'article : la production visitee `y` si elle est localement atteignable pour cet intervalle,
c'est-a-dire dans
    [P_min u_{a,b,t}, ȳ_{a,b,t}]                                   si a >= 1,
    [u_{a,b,t} max{P_min, y_{j,0} - R^D t}, ȳ_{a,b,t}]             si a <= 0,
sinon la production maximale atteignable ȳ (`reference_output`). Un support local est valide en
n'importe quel point : ce choix ne change que la qualite de la coupe aux autres plannings, ou cet
intervalle serait choisi. A utiliser pour les intervalles NON choisis ; un intervalle choisi est
linearise en `y` (la coupe doit etre serree au point visite).
"""
function linearization_point(unit, a, b, t, T, y; tol = 1e-8)
    ybar = reference_output(unit, a, b, t, T)
    lower =
        !is_on_interval(a, b, t) ? 0.0 :
        a >= 1 ? unit.MinPower :
        max(unit.MinPower, unit.InitialPower - unit.DeltaRampDown * t)
    return lower - tol <= y <= ybar + tol ? y : ybar
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
