#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    commitment_duals(params, instance, t)

Coefficients locaux sur x de la valeur de l'etape `t` (3-bin) : duaux des parametres du
planning, places a leur position dans le vecteur plat de `Schedule.x`. Les coordonnees
qui n'apparaissent pas dans les contraintes de l'etape restent nulles.
"""
function commitment_duals(params::StageParameters, instance::Instance, t::Int)
    τ = zeros(commitment_dimension(instance))
    for i in 1:instance.N
        τ[u_index(instance, i, t-1)] += JuMP.dual(JuMP.FixRef(params.u_prev[i]))
        τ[u_index(instance, i, t)] += JuMP.dual(JuMP.FixRef(params.u[i]))
        τ[v_index(instance, i, t)] += JuMP.dual(JuMP.FixRef(params.v[i]))
        τ[w_index(instance, i, t)] += JuMP.dual(JuMP.FixRef(params.w[i]))
        for (h, p) in enumerate(params.w_future[i])
            τ[w_index(instance, i, t+h)] += JuMP.dual(JuMP.FixRef(p))
        end
    end
    return τ
end

# Mettre a `false` pour lire les duaux des coupes un par un avec `JuMP.dual` (repli generique, plus lent).
const BULK_CUT_DUALS = Ref(true)

# Mettre a `true` pour comparer, a chaque lecture groupee, avec `JuMP.dual` et s'arreter au moindre ecart.
const CHECK_BULK_CUT_DUALS = Ref(false)

# Optimiseur Gurobi interne du sous-probleme, ou `nothing` si le solveur n'est pas Gurobi.
function _gurobi_optimizer(subproblem::JuMP.Model)
    inner = JuMP.unsafe_backend(subproblem)
    return inner isa Gurobi.Optimizer ? inner : nothing
end

# Ligne (numerotee depuis 0) de la coupe dans le modele Gurobi.
_cut_row(inner::Gurobi.Optimizer, cut) =
    Cint(Gurobi._info(inner, JuMP.optimizer_index(cut.constraint_ref)).row - 1)

# Lignes des coupes de `node` dans le modele Gurobi, calculees une seule fois par coupe puis gardees dans
# `node.ext` : retrouver une ligne a partir d'une coupe coute bien plus cher que la lecture des duaux.
# ATTENTION : si le solveur recopie tout le modele (rattrapage numerique : `reset_optimizer`, qui vide ce
# cache ; ou un `Parameter` JuMP modifie, ce que le code evite), il range les lignes dans un autre ordre
# (les coupes anciennes se retrouvent regroupees, les nouvelles en fin de modele). Filet de securite : si la
# derniere ligne gardee n'est plus la bonne, tout le cache est refait.
function _cut_rows(node::Node, inner::Gurobi.Optimizer)
    rows = get!(() -> Cint[], node.ext, :cut_rows)::Vector{Cint}
    if !isempty(rows) && _cut_row(inner, node.cuts[length(rows)]) != rows[end]
        empty!(rows)
    end
    for k in (length(rows)+1):length(node.cuts)
        push!(rows, _cut_row(inner, node.cuts[k]))
    end
    return rows
end

"""
    _cut_duals(node)

Duaux (σ) des coupes de `node` au dernier probleme resolu, dans l'ordre de `node.cuts`. Avec Gurobi, ils
sont lus en un seul appel (le vecteur `Pi` de toutes les lignes) au lieu d'un `JuMP.dual` par coupe,
dont le cout (environ 0,4 µs et 300 octets chacun, et plus a grand nombre de coupes) dominait la passe
arriere ; seuls quelques pourcents d'entre eux sont non nuls. Meme valeur que `JuMP.dual`, signe compris
(Gurobi.jl renvoie `Pi` en minimisation et `-Pi` en maximisation). Autre solveur, certificat
d'infaisabilite ou `BULK_CUT_DUALS[] = false` : lecture coupe par coupe.
"""
function _cut_duals(node::Node)
    inner = BULK_CUT_DUALS[] ? _gurobi_optimizer(node.subproblem) : nothing
    if inner === nothing || inner.has_infeasibility_cert
        return [JuMP.dual(cut.constraint_ref) for cut in node.cuts]
    end
    rows = _cut_rows(node, inner)
    numconstrs = Ref{Cint}()
    Gurobi._check_ret(inner, Gurobi.GRBgetintattr(inner, "NumConstrs", numconstrs))
    pi = get!(() -> Cdouble[], node.ext, :pi_buffer)::Vector{Cdouble}
    length(pi) < numconstrs[] && resize!(pi, numconstrs[])
    Gurobi._check_ret(inner, Gurobi.GRBgetdblattrarray(inner, "Pi", 0, numconstrs[], pi))
    sign = MOI.get(inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE ? 1.0 : -1.0
    duals = [sign * pi[row+1] for row in rows]
    if CHECK_BULK_CUT_DUALS[]
        reference = [JuMP.dual(cut.constraint_ref) for cut in node.cuts]
        duals == reference ||
            error("duaux des coupes du noeud $(node.index) : la lecture groupee differe de JuMP.dual")
    end
    return duals
end

"""
    get_dual_solution(node)

Multiplicateurs du sous-probleme d'etape `node`, resolu au point courant (`StageDuals`). En
formulation intervalle, les pentes en y (`β`) et en x (`τ`) sont calculees en forme close
(`local_support`) et non lues dans les duaux : ces deux champs restent vides, ce qui evite un appel
`JuMP.dual` (environ 0,4 µs) par unite et par resolution.
"""
function get_dual_solution(node::Node)
    if !_has_dual_solution(node)
        model = node.subproblem.ext[:MSUC_policy_graph]
        attempt_numerical_recovery(model, node; require_dual = true)
    end
    model = get_policy_graph(node.subproblem)
    extended = model.method == ExtendedFormulation
    β = extended ? Float64[] : [JuMP.dual(JuMP.FixRef(s.in)) for s in node.power]
    τ =
        extended ? Float64[] :
        commitment_duals(node.planning, model.instance, node.index - 1)
    return StageDuals(
        _cut_duals(node),
        JuMP.dual.(node.demand_constraint),
        β,
        [JuMP.dual(JuMP.FixRef(s.in)) for s in node.zeta],
        τ,
    )
end
