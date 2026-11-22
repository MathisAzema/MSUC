# Root commitment problem. Power and ζ are copied through to pass the initial state to stage 1.
function _root_problem!(subproblem, instance, power, ζ, basis)
    N, T = instance.N, instance.TimeHorizon
    units = instance.Thermalunits
    @variable(subproblem, u[i = 1:N, t = 0:T], Bin)
    @variable(subproblem, v[i = 1:N, t = 1:T], Bin)
    @variable(subproblem, w[i = 1:N, t = 1:T], Bin)
    add_commitment_constraints!(subproblem, instance, u, v, w)
    set_stage_objective(subproblem, commitment_cost(instance, u, v, w))
    @constraint(subproblem, [i in 1:N], power[i].out == power[i].in)
    @constraint(subproblem, [i in eachindex(ζ)], ζ[i].out == ζ[i].in)

    gamma = Dict{Tuple{Int,Int,Int},JuMP.VariableRef}()
    if basis === :intervals
        # gamma_{j,a,b} >= 0 linked to x by (18c).
        for i in 1:N, (a, b) in units[i].intervals
            gamma[i, a, b] = @variable(subproblem, lower_bound = 0.0, base_name = "gamma_$(i)_$(a)_$(b)")
        end
        @constraint(subproblem, [unit in units, t in 1:T],
            sum(gamma[unit.name, a, b] for (a, b) in unit.intervals if a <= t < b) == u[unit.name, t])
        @constraint(subproblem, [unit in units, t in 1:T],
            sum(gamma[unit.name, a, b] for (a, b) in unit.intervals if a == t) == v[unit.name, t])
        @constraint(subproblem, [unit in units, t in 1:T],
            sum(gamma[unit.name, a, b] for (a, b) in unit.intervals if b == t) == w[unit.name, t])
    end
    return RootVariables(u, v, w, gamma)
end
