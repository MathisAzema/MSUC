"""
    solve_first_stage_problem(model, initial_state; time_limit = nothing)

Solve the root MIP (Algorithm 2, lines 4-5): choose the schedule x^k (and gamma^k with the interval
formulation) given the current future-cost approximation. Fixing x^k in the dispatch stages is left to
`one_iteration`.

`time_limit` (seconds) caps this MIP; a later call without it removes the cap. If the limit is hit, the
dual bound stays valid and `incumbent` is the best schedule found, or `nothing` if there is none.

Returns `(bound, incumbent)`: `bound` is the lower bound L^k, i.e. the MIP *dual* bound (with a nonzero
`MIPGap` the incumbent value is not a valid lower bound); `incumbent` is the objective value of x^k.
"""
function solve_first_stage_problem(
    model::PolicyGraph,
    initial_state;
    time_limit::Union{Real,Nothing} = nothing,
)
    root = model[model.first_node]
    noise = root.noise_terms[1].term
    # Set before solving and never reset afterwards: changing a solver parameter invalidates the
    # solution that `one_iteration` reads next (`extract_schedule`).
    if time_limit !== nothing
        JuMP.set_time_limit_sec(root.subproblem, max(Float64(time_limit), 0.0))
        model.ext[:root_time_limit_set] = true
    elseif get(model.ext, :root_time_limit_set, false)
        JuMP.set_time_limit_sec(root.subproblem, nothing)
        model.ext[:root_time_limit_set] = false
    end
    result = solve_subproblem(model, root, initial_state, noise)
    if result === nothing
        bound = try
            JuMP.objective_bound(root.subproblem)
        catch
            -Inf
        end
        return (bound = bound, incumbent = nothing)
    end
    return (
        bound = JuMP.objective_bound(root.subproblem),
        incumbent = JuMP.objective_value(root.subproblem),
    )
end

"""
    first_stage_value(model, schedule)

Objective of the root MIP at `schedule` with the current approximation (commitment cost plus the
maximum of the root cuts), without solving it.
"""
function first_stage_value(model::PolicyGraph, schedule::Schedule)
    root = model[model.first_node]
    return schedule_commitment_cost(model.instance, schedule) +
           maximum(root_cut_value(model, cut, schedule) for cut in root.cuts)
end

"""
    reusable_incumbent(model)

Return `(bound, incumbent)` if the previous schedule x^k, evaluated with the new approximation, is within
the MIP relative gap (as defined by Gurobi) of the best bound, so the root MIP can be skipped; `nothing`
otherwise. Valid because the approximation never decreases, so the best bound also bounds the new MIP.
"""
function reusable_incumbent(model::PolicyGraph)
    model.schedule === nothing && return nothing
    bound = get(model.ext, :best_bound, -Inf)
    isfinite(bound) || return nothing
    value = first_stage_value(model, model.schedule)
    gap = JuMP.get_optimizer_attribute(model[model.first_node].subproblem, MOI.RelativeGapTolerance())
    value - bound <= gap * abs(value) + 1e-9 || return nothing
    return (bound = bound, incumbent = value)
end
