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
        τ[u_index(instance, i, t-1)] += JuMP.dual(JuMP.ParameterRef(params.u_prev[i]))
        τ[u_index(instance, i, t)] += JuMP.dual(JuMP.ParameterRef(params.u[i]))
        τ[v_index(instance, i, t)] += JuMP.dual(JuMP.ParameterRef(params.v[i]))
        τ[w_index(instance, i, t)] += JuMP.dual(JuMP.ParameterRef(params.w[i]))
        for (h, p) in enumerate(params.w_future[i])
            τ[w_index(instance, i, t+h)] += JuMP.dual(JuMP.ParameterRef(p))
        end
    end
    return τ
end

"""
    get_dual_solution(node)

Multiplicateurs du sous-probleme d'etape `node`, resolu au point courant (`StageDuals`).
"""
function get_dual_solution(node::Node)
    if !_has_dual_solution(node)
        model = node.subproblem.ext[:MSUC_policy_graph]
        attempt_numerical_recovery(model, node; require_dual = true)
    end
    model = get_policy_graph(node.subproblem)
    τ =
        model.method == ExtendedFormulation ? Float64[] :
        commitment_duals(node.planning, model.instance, node.index - 1)
    return StageDuals(
        [JuMP.dual(cut.constraint_ref) for cut in node.cuts],
        JuMP.dual.(node.demand_constraint),
        [JuMP.dual(JuMP.FixRef(s.in)) for s in node.power],
        [JuMP.dual(JuMP.FixRef(s.in)) for s in node.zeta],
        τ,
    )
end
