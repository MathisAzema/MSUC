# Deterministic equivalent of the scenario tree: the exact optimum of the multistage problem (1),
# with nonanticipativity (1j). Used as a reference for the SDDP bounds. The tree has
# sum_{t=0}^T S^t nodes, so it is tractable only for small S and T; with an instance built with
# S = 1 it reduces to a path (the deterministic problem).

"""
    scenario_tree(instance, T) -> (parent, innovation, probability, eta)

ARMA scenario tree, indexed by `(t, k)` with `k in 1:S^t`. For each stage `t in 0:T` it returns the
parent node, the innovation index, the node probability and the relative error `eta` of each bus.
Since `eta` is deterministic along a path, recursion (1g) is evaluated numerically (no `zeta` variable).
"""
function scenario_tree(instance::Instance, T::Int)
    a = instance.arma
    S = length(a.support)
    nb = size(a.S_eta, 1)
    parent = [zeros(Int, S^t) for t in 0:T]
    innovation = [zeros(Int, S^t) for t in 0:T]
    probability = [zeros(S^t) for t in 0:T]
    zeta = [[zeros(zeta_dimension(a)) for _ in 1:S^t] for t in 0:T]
    eta = [[zeros(nb) for _ in 1:S^t] for t in 0:T]

    probability[1][1] = 1.0
    zeta[1][1] = copy(a.zeta0)
    eta[1][1] = a.S_eta * a.zeta0
    for t in 1:T
        for k in 1:S^t
            p = div(k - 1, S) + 1
            s = mod(k - 1, S) + 1
            parent[t+1][k] = p
            innovation[t+1][k] = s
            probability[t+1][k] = probability[t][p] * a.probability[s]
            zeta[t+1][k] = a.Psi * zeta[t][p] + a.E * a.support[s]
            eta[t+1][k] = a.S_eta * zeta[t+1][k]
        end
    end
    return parent, innovation, probability, eta
end

function thermal_unit_commit_constraints(model, instance)
    is_on = model[:is_on]
    start_up = model[:start_up]
    start_down = model[:start_down]
    add_commitment_constraints!(model, instance, is_on, start_up, start_down)
    @constraint(model, model[:thermal_fixed_cost] >= commitment_cost(instance, is_on, start_up, start_down))
    return
end

"""
    multistage_extensive(instance; force, silent, gap, timelimit, res_gamma)

Solve the scenario-tree deterministic equivalent (exact optimum of problem (1)). The commitment is
shared by the whole tree and dispatch is indexed by tree node, which encodes (1j). The shutdown
feasibility lines (44)/(46) are omitted: over the full horizon the ramps imply them, so the
reference is independent of them.

Returns `(value, bound, time, schedule)`, or `nothing` on failure. `res_gamma` fixes the
commitment to a given schedule, e.g. to evaluate one produced by SDDP.
"""
function multistage_extensive(
    instance::Instance;
    force::Float64 = 1.0,
    silent::Bool = true,
    gap::Float64 = 1e-9,
    timelimit::Int = 300,
    res_gamma = nothing,
)
    T = instance.TimeHorizon
    units = instance.Thermalunits
    names = [u.name for u in units]
    Next = instance.Next
    Buses = 1:size(Next)[1]
    Lines = values(instance.Lines)
    S = length(instance.arma.support)
    _, _, probability, eta = scenario_tree(instance, T)
    nodes_at(t) = 1:S^t
    parent_of(t, k) = div(k - 1, S) + 1

    model = Model(instance.optimizer)
    silent && set_silent(model)

    @variable(model, is_on[j in names, t in 0:T], Bin)
    @variable(model, start_up[j in names, t in 1:T], Bin)
    @variable(model, start_down[j in names, t in 1:T], Bin)
    @variable(model, thermal_fixed_cost >= 0)
    thermal_unit_commit_constraints(model, instance)
    if res_gamma !== nothing
        @constraint(model, [u in units, t in 1:T],
            sum(res_gamma[u.name, a, b] for (a, b) in u.intervals if a <= t < b) == is_on[u.name, t])
        @constraint(model, [u in units, t in 1:T],
            sum(res_gamma[u.name, a, b] for (a, b) in u.intervals if a == t) == start_up[u.name, t])
        @constraint(model, [u in units, t in 1:T],
            sum(res_gamma[u.name, a, b] for (a, b) in u.intervals if b == t) == start_down[u.name, t])
    end

    power = Dict((j, t, k) => @variable(model, lower_bound = 0.0)
                 for j in names for t in 0:T for k in nodes_at(t))
    shed = Dict((b, t, k) => @variable(model, lower_bound = 0.0)
                for b in Buses for t in 1:T for k in nodes_at(t))
    curt = Dict((b, t, k) => @variable(model, lower_bound = 0.0)
                for b in Buses for t in 1:T for k in nodes_at(t))
    θ = Dict((b, t, k) => @variable(model) for b in Buses for t in 1:T for k in nodes_at(t))
    flow = Dict((b, bp, t, k) => @variable(model)
                for b in Buses for bp in Next[b] for t in 1:T for k in nodes_at(t))

    for u in units
        @constraint(model, power[(u.name, 0, 1)] == u.InitialPower)
    end
    for t in 1:T, k in nodes_at(t)
        kp = parent_of(t, k)
        for u in units
            j = u.name
            add_unit_dynamics!(model, u, power[(j, t, k)], power[(j, t - 1, kp)],
                               is_on[j, t], is_on[j, t-1], start_up[j, t], start_down[j, t])
        end
        for line in Lines
            add_line_constraints!(model, line, flow[(line.b1, line.b2, t, k)],
                                  θ[(line.b1, t, k)], θ[(line.b2, t, k)])
        end
        for b in Buses
            @constraint(model,
                sum(power[(u.name, t, k)] for u in units if u.Bus == b) - curt[(b, t, k)] +
                shed[(b, t, k)] ==
                instance.Demandbus[b][t] * (1 + force * eta[t+1][k][b]) +
                sum(flow[(b, bp, t, k)] for bp in Next[b]))
        end
    end

    @objective(model, Min,
        thermal_fixed_cost + sum(
            probability[t+1][k] * (
                sum(u.LinearTerm * power[(u.name, t, k)] for u in units) + sum(
                    SHEDDING_COST * shed[(b, t, k)] + CURTAILEMENT_COST * curt[(b, t, k)]
                    for b in Buses)
            ) for t in 1:T for k in nodes_at(t)))

    set_optimizer_attribute(model, "TimeLimit", timelimit)
    set_optimizer_attribute(model, "MIPGap", gap)
    start = time()
    optimize!(model)
    elapsed = time() - start
    primal_status(model) == MOI.FEASIBLE_POINT || return nothing
    schedule = [value.(is_on), value.(start_up), value.(start_down)]
    return objective_value(model), objective_bound(model), elapsed, schedule
end
