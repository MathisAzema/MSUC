#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Etat porte d'une etape a l'autre : la production y et l'etat d'incertitude zeta,
# sous la forme `(power = [...], zeta = [...])`.
function initial_state(instance::Instance)
    return (
        power = [unit.InitialPower for unit in instance.Thermalunits],
        zeta = copy(instance.arma.zeta0),
    )
end

function set_incoming_state(node::Node, state)
    JuMP.fix.([s.in for s in node.power], state.power)
    JuMP.fix.([s.in for s in node.zeta], state.zeta)
    return
end

# Valeur d'une variable ramenee dans ses bornes (le solveur tolere de petits depassements).
function _clamped_value(var::JuMP.VariableRef)
    value = JuMP.value(var)
    if JuMP.has_upper_bound(var)
        value = min(value, JuMP.upper_bound(var))
    end
    if JuMP.has_lower_bound(var)
        value = max(value, JuMP.lower_bound(var))
    end
    return value
end

function get_outgoing_state(node::Node)
    return (
        power = [_clamped_value(s.out) for s in node.power],
        zeta = [_clamped_value(s.out) for s in node.zeta],
    )
end

function set_objective(node::Node)
    if !node.stage_objective_set
        JuMP.set_objective(
            node.subproblem,
            JuMP.objective_sense(node.subproblem),
            @expression(
                node.subproblem,
                node.stage_objective + bellman_term(node.bellman_function)
            )
        )
    end
    node.stage_objective_set = true
    return
end

_value(x::Real) = x

_value(x) = JuMP.value(x)

stage_objective_value(::Node, x::Union{GenericVariableRef,Real}) = _value(x)

function stage_objective_value(node::Node, stage_objective)
    theta = bellman_term(node.bellman_function)
    return JuMP.objective_value(node.subproblem) - JuMP.value(theta)
end

function write_subproblem_to_file(
    node::Node,
    filename::String;
    throw_error::Bool = false,
)
    model = MOI.FileFormats.Model(; filename = filename)
    MOI.copy_to(model, JuMP.backend(node.subproblem))
    MOI.write_to_file(model, filename)
    if throw_error
        error(
            "Unable to retrieve solution from node $(node.index).\n\n",
            "  Termination status : $(JuMP.termination_status(node.subproblem))\n",
            "  Primal status      : $(JuMP.primal_status(node.subproblem))\n",
            "  Dual status        : $(JuMP.dual_status(node.subproblem)).\n\n",
            "The current subproblem was written to `$(filename)`.\n\n",
            "There are two common causes of this error:\n",
            "  1) you have a mistake in your formulation, or you violated\n",
            "     the assumption of relatively complete recourse\n",
            "  2) the solver encountered numerical issues\n\n",
            "See https://odow.github.io/MSUC.jl/stable/tutorial/warnings/ for more information.",
        )
    end
    return
end

function parameterize(node::Node, noise)
    node.parameterize(noise)
    set_objective(node)
    return
end

function _has_primal_solution(node::Node)
    status = JuMP.primal_status(node.subproblem)
    return status in (JuMP.FEASIBLE_POINT, JuMP.NEARLY_FEASIBLE_POINT)
end

function _has_dual_solution(node::Node)
    status = JuMP.dual_status(node.subproblem)
    return status in (JuMP.FEASIBLE_POINT, JuMP.NEARLY_FEASIBLE_POINT)
end

function set_numerical_difficulty_callback(
    model::PolicyGraph,
    callback::Function,
)
    model.ext[:numerical_difficulty_callback] = callback
    return
end

function attempt_numerical_recovery(
    model::PolicyGraph,
    node::Node;
    require_dual::Bool = false,
)
    model.ext[:numerical_issue] = true
    callback = get(
        model.ext,
        :numerical_difficulty_callback,
        default_numerical_difficulty_callback,
    )
    callback(model, node; require_dual)
    missing_dual_solution = require_dual && !_has_dual_solution(node)
    if !_has_primal_solution(node) || missing_dual_solution
        write_subproblem_to_file(
            node,
            "subproblem_$(node.index).mof.json";
            throw_error = true,
        )
    end
    return
end

function default_numerical_difficulty_callback(
    model::PolicyGraph,
    node::Node;
    kwargs...,
)
    if JuMP.mode(node.subproblem) == JuMP.DIRECT
        @warn(
            "Unable to recover in direct mode! Remove `direct = true` when " *
            "creating the policy graph."
        )
        return
    end
    MOI.Utilities.reset_optimizer(node.subproblem)
    delete!(node.ext, :cut_rows)    # les lignes des coupes dans le modele Gurobi ont pu changer
    optimize!(node.subproblem)
    return
end

function solve_subproblem(
    model::PolicyGraph,
    node::Node,
    state,
    noise;
    compute_duals::Bool = false,
)
    set_incoming_state(node, state)
    parameterize(node, noise)
    start = time()
    JuMP.optimize!(node.subproblem)
    push!(node.computation_time, time() - start)
    model.ext[:total_solves] = get(model.ext, :total_solves, 0) + 1
    if JuMP.primal_status(node.subproblem) == JuMP.MOI.INTERRUPTED
        throw(InterruptException())
    end
    if !_has_primal_solution(node) && JuMP.termination_status(node.subproblem) == MOI.TIME_LIMIT
        # Limite de temps atteinte sans solution realisable : seul le MIP racine peut en avoir une
        # (`solve_first_stage_problem`). Pas de recuperation numerique, qui le relancerait : l'appelant decide.
        return nothing
    end
    if !_has_primal_solution(node)
        attempt_numerical_recovery(model, node)
    end
    state = get_outgoing_state(node)
    stage_objective = stage_objective_value(node, node.stage_objective)
    duals = nothing
    if compute_duals
        TimerOutputs.@timeit model.timer_output "get_dual_solution" begin
            duals = get_dual_solution(node)
        end
    end
    objective = JuMP.objective_value(node.subproblem)
    return (
        state = state,
        duals = duals,
        objective = objective,
        stage_objective = stage_objective,
    )
end
