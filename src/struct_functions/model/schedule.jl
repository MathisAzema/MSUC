# =============================================================================
#  Le planning d'engagement x = (u, v, w) est decide a la racine, puis reste FIXE
#  pendant toute l'iteration : ce n'est pas un etat du programme dynamique mais un
#  parametre des etapes de dispatch (algorithme 1 et 2 : x^k, gamma^k fixes). Les
#  seuls etats sont la production y_t et l'etat d'incertitude zeta_t.
#
#  Trois objets :
#    RootVariables     variables de decision du probleme racine (MIP)
#    Schedule          valeurs du planning x^k lues a la racine apres resolution
#    StageParameters   parametres JuMP des etapes de dispatch, fixes a x^k
# =============================================================================

"""
    RootVariables

Variables de decision du noeud racine : le planning `u` (sur `0:T`), `v` et `w` (sur
`1:T`), indexes `[unit.name, t]`, et, pour la formulation intervalle, les `gamma[j, a, b]`
(vide en 3-bin).
"""
struct RootVariables
    u::Any
    v::Any
    w::Any
    gamma::Dict{Tuple{Int,Int,Int},JuMP.VariableRef}
end

"""
    StageParameters

Parametres du planning dans le sous-probleme de l'etape `t` : le statut `u_prev` (t-1) et
`u` (t), le demarrage `v` et l'arret `w` a `t`, plus selon la formulation
  * 3-bin      : `w_future[j][h]` = w_{t+h}, necessaires aux lignes de faisabilite d'arret
                 (44), dont les duaux portent les coefficients de la coupe sur x ;
  * intervalle : `cap[j]`, l'enveloppe d'arret (46) de l'intervalle actif.
"""
struct StageParameters
    u_prev::Vector{JuMP.VariableRef}
    u::Vector{JuMP.VariableRef}
    v::Vector{JuMP.VariableRef}
    w::Vector{JuMP.VariableRef}
    w_future::Vector{Vector{JuMP.VariableRef}}
    cap::Vector{JuMP.VariableRef}
end

"""
    Schedule

Planning x^k = (u, v, w) lu a la racine, arrondi a {0, 1}. `u[i, t+1]` vaut u_{i,t}
(t in 0:T), `v[i, t]` et `w[i, t]` valent v_{i,t} et w_{i,t} (t in 1:T). `selected` liste
les intervalles [a:b] choisis (gamma = 1), vide en 3-bin. `x` est le vecteur plat
(u, v, w) : sa i-eme coordonnee est celle de `u_index`, `v_index`, `w_index`.
"""
struct Schedule
    u::Matrix{Float64}
    v::Matrix{Float64}
    w::Matrix{Float64}
    selected::Vector{Tuple{Int,Int,Int}}
    selected_set::Set{Tuple{Int,Int,Int}}
    x::Vector{Float64}
end

function Schedule(u, v, w, selected)
    return Schedule(
        u,
        v,
        w,
        selected,
        Set(selected),
        vcat(vec(u), vec(v), vec(w)),
    )
end
