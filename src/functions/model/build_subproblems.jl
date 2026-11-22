# Stage subproblems of both formulations. Node 1 is the (only nonconvex) root commitment problem,
# nodes t+1 >= 2 are the dispatch problem of stage t. States are the power y and the uncertainty
# state zeta; the schedule x = (u, v, w) is a fixed parameter of the dispatch stages (see
# schedule.jl). The formulations differ only in the shutdown feasibility rows: 3-bin uses lines
# (44d), one per lag h, on parameters w_{t+h}; the interval formulation uses the single line (45),
# `power <= cap`, with no dual on x (the closed-form local support in thermal_unit.jl replaces it).

# States are declared at every node, root included, so the state space is the same everywhere.
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

# `basis` (`:bins` or `:intervals`) is the schedule basis in the dispatch stages, hence the cut type.
# `root_basis` (default `basis`) is the root problem's; with `:intervals` it adds the gamma variables.
function _stage_builder!(instance, subproblem, node_index, force, basis; root_basis = basis)
    power, ζ = _declare_states!(subproblem, instance)
    node = get_node(subproblem)
    if node_index == 1
        node.planning = _root_problem!(subproblem, instance, power, ζ, root_basis)
    else
        t = node_index - 1
        node.planning = _stage_parameters!(subproblem, instance, power, t, basis)
        node.demand_constraint = _dispatch_common!(subproblem, instance, force, power, ζ, t)
    end
    return
end

"""
    subproblem_builder_UC_def(instance, force, subproblem, node)

Subproblem of the reference 3-bin formulation (Sect. 2).
"""
function subproblem_builder_UC_def(instance::MSUC.Instance, force::Float64, subproblem::Model, node::Int)
    return _stage_builder!(instance, subproblem, node, force, :bins)
end

"""
    subproblem_builder_UC_def_gamma(instance, force, subproblem, node)

3-bin formulation whose root problem also carries the gamma_{j,a,b} variables linked by (18c).
Dispatch stages and cuts (12) are the 3-bin ones; use with `SDDPFormulation`. The gammas do not
tighten the root relaxation in x-space (same LP and MIP values with the same cuts), so this only
changes the model size.
"""
function subproblem_builder_UC_def_gamma(instance::MSUC.Instance, force::Float64, subproblem::Model, node::Int)
    return _stage_builder!(instance, subproblem, node, force, :bins; root_basis = :intervals)
end

"""
    subproblem_builder_UC_extended(instance, force, subproblem, node)

Subproblem of the interval formulation (Sect. 3): gamma variables (Def. 1) at the root, shutdown
line (45) in the dispatch stages.
"""
function subproblem_builder_UC_extended(instance::MSUC.Instance, force::Float64, subproblem::Model, node::Int)
    return _stage_builder!(instance, subproblem, node, force, :intervals)
end
