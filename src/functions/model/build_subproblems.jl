# =============================================================================
#  Sous-problemes d'etape des deux formulations.
#
#  Noeud 1 (racine)  : le probleme d'engagement, (7) en 3-bin et (19) en intervalle.
#                      C'est le seul noeud non convexe.
#  Noeuds t+1 >= 2   : le probleme de dispatch de l'etape t.
#
#  Les ETATS portes d'un noeud a l'autre sont seulement
#    y_{j,t}   production physique de l'unite j        (`power`)
#    zeta_t    etat d'incertitude de la recursion ARMA   (`ζ`)
#  Le planning x = (u, v, w) est decide a la racine puis fixe : c'est un PARAMETRE des
#  etapes de dispatch (cf. schedule.jl), pas un etat. Les deux formulations partagent
#  tout le sous-probleme d'etape, sauf la faisabilite d'arret :
#    3-bin      lignes (44), une par decalage h, ecrites sur les parametres w_{t+h} :
#               leurs duaux portent les coefficients de la coupe sur x ;
#    intervalle une ligne (46), power <= cap, ou `cap` est l'enveloppe d'arret de
#               l'intervalle actif : aucun dual sur x, le support local en forme close
#               (Thermal_unit.jl) tient lieu de coefficient.
#  Chaque builder renseigne directement les champs du noeud (`power`, `zeta`,
#  `demand_constraint`, `planning`).
# =============================================================================

# Etats : production et etat d'incertitude. Declares a TOUS les noeuds, racine comprise,
# pour que l'espace d'etat soit le meme partout.
function _declare_states!(subproblem, instance)
    N = instance.N
    units = instance.Thermalunits
    nz = MSUC.zeta_dimension(instance.arma)
    @variable(
        subproblem,
        0.0 <= power[i = 1:N] <= units[i].MaxPower,
        MSUC.State,
        initial_value = units[i].InitialPower
    )
    @variable(subproblem, ζ[i = 1:nz], MSUC.State, initial_value = instance.arma.zeta0[i])
    node = get_node(subproblem)
    node.power = collect(power)
    node.zeta = collect(ζ)
    return node.power, node.zeta
end

function _stage_builder!(instance, subproblem, node_index, force, basis)
    power, ζ = _declare_states!(subproblem, instance)
    node = get_node(subproblem)
    if node_index == 1
        node.planning = _root_problem!(subproblem, instance, power, ζ, basis)
    else
        t = node_index - 1
        node.planning = _stage_parameters!(subproblem, instance, power, t, basis)
        node.demand_constraint = _dispatch_common!(subproblem, instance, force, power, ζ, t)
    end
    return
end

"""
    subproblem_builder_UC_def(instance, force, subproblem, node)

Sous-probleme de la formulation 3-bin de reference (Sect. 2). La faisabilite d'arret est
imposee par les lignes (44), une par decalage h : le planning 3-bin ne porte pas l'instant
d'arret, il faut donc enumerer les decalages.
"""
function subproblem_builder_UC_def(instance::MSUC.Instance, force::Float64, subproblem::Model, node::Int)
    return _stage_builder!(instance, subproblem, node, force, :bins)
end

"""
    subproblem_builder_UC_extended(instance, force, subproblem, node)

Sous-probleme de la formulation intervalle (Sect. 3). A la racine, les variables
gamma_{j,a,b} (Def. 1) sont reliees au planning par (19c). Aux etapes de dispatch, la
faisabilite d'arret est la ligne (46) : l'instant d'arret est explicite, aucun decalage a
enumerer, et le sous-probleme n'a plus aucune variable gamma.
"""
function subproblem_builder_UC_extended(instance::MSUC.Instance, force::Float64, subproblem::Model, node::Int)
    return _stage_builder!(instance, subproblem, node, force, :intervals)
end
