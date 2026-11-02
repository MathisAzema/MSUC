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
#    ytilde^k            point de linearisation (y ou ybar, conditions (i)-(ii))      -> linearization_point
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
    ramp_steps(unit, rate)

T^R_j de l'article : ⌈(P_max - P_min) / rate⌉, nombre de pas pour parcourir la plage [P_min, P_max] en
montee (`rate` = R^U, T^RU) ou en descente (`rate` = R^D, T^RD).
"""
ramp_steps(unit, rate) = ceil(Int, (unit.MaxPower - unit.MinPower) / rate)

"""
    linearization_point(unit, a, b, t, T, y, selected, chosen)

Production ỹ^k_{j,a,b,t} de l'intervalle [a:b] au pas t (etat entrant de l'etape s = t + 1) ou l'on linearise
le probleme local : la production visitee `y` si
  (i)  l'intervalle est choisi (`selected`, γ = 1), car la coupe doit etre serree au point visite ; ou
  (ii) l'intervalle est assez long, b - a >= T^RU + T^RD, et il existe un intervalle choisi [c:d] de la meme
       unite (`chosen`, liste des (c, d)) tel que s appartienne a
           [max(a, c) + (1 + T^RU) 1_{a≠c} : min(b, d) - T^RD 1_{b≠d}],
       car `y`, atteignable depuis [c:d], l'est alors aussi depuis [a:b] ;
et la production maximale atteignable ȳ (`reference_output`) sinon. Un support local est valide en
n'importe quel point : ce choix ne change que la qualite de la coupe aux autres plannings, ou cet
intervalle serait choisi.

Pour un intervalle initial non choisi (a <= 0, depart a y_{j,0}) et un intervalle choisi [c:d] different de
lui (c ≠ a), (ii) exige aussi t >= ⌈(y_{j,0} - P_min) / R^D⌉ : la production de [a:b] ne peut pas descendre
plus vite que R^D depuis y_{j,0}, elle reste donc >= max(P_min, y_{j,0} - R^D t), et un `y` plus bas, atteignable
depuis [c:d], ne l'est pas depuis [a:b]. Si c = a, les deux intervalles ont la meme borne basse.

Les marges sont les plus petites qui garantissent l'atteignabilite (conventions du code : u = 1 sur
[a : b-1], S^U = S^D = P_min). Quand les debuts different, il faut que le plus tardif ait fini sa montee
pour que la plage soit complete : atteindre P_max demande T^RU pas, et l'etat entrant de l'etape s est
y_{s-1}, d'ou s >= a + T^RU + 1. A la fin, le dernier pas allume est b - 1 et la descente demande T^RD
pas, d'ou s <= b - T^RD. Quand les bornes coincident, les cones sont identiques.

Dans le modele du code (S^D = P_min), la condition sur les intervalles initiaux est toujours satisfaite par
les plannings admissibles : un intervalle initial [a:b] n'existe que pour b >= 1 + ⌈(y_{j,0} - P_min) / R^D⌉
(`set_intervals`), donc tout redemarrage choisi commence apres la fin de la descente. Elle ne change alors
aucun resultat ; elle protege une generalisation a S^D > P_min, ou une unite initialement allumee peut
s'arreter avant d'etre redescendue a P_min.
"""
function linearization_point(unit, a, b, t, T, y, selected::Bool, chosen)
    selected && return y
    t_up = ramp_steps(unit, unit.DeltaRampUp)
    t_down = ramp_steps(unit, unit.DeltaRampDown)
    if b - a >= t_up + t_down
        s = t + 1
        # pas necessaires a un intervalle initial pour que sa borne basse retombe a P_min
        t_initial = a <= 0 ? ceil(Int, max(0.0, unit.InitialPower - unit.MinPower) / unit.DeltaRampDown) : 0
        for (c, d) in chosen
            a <= 0 && c != a && t < t_initial && continue
            lower = max(a, c) + (a != c ? 1 + t_up : 0)
            upper = min(b, d) - (b != d ? t_down : 0)
            lower <= s <= upper && return y
        end
    end
    return reference_output(unit, a, b, t, T)
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
