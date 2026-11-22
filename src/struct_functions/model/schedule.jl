"""
    RootVariables

Root decision variables: `u` (on `0:T`), `v` and `w` (on `1:T`) indexed `[unit.name, t]`,
and `gamma[j, a, b]` for the interval formulation (empty for 3-bin).
"""
struct RootVariables
    u::Any
    v::Any
    w::Any
    gamma::Dict{Tuple{Int,Int,Int},JuMP.VariableRef}
end

"""
    StageParameters

Schedule values fixed (`JuMP.fix`) in the stage-`t` subproblem: `u_prev`, `u`, `v`, `w`, plus
  * 3-bin:    `w_future[j][h]` = w_{t+h}, used by the shutdown-feasibility rows (44d);
  * interval: `cap[j]`, the shutdown envelope (45) of the active interval.
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

Schedule x^k = (u, v, w) read at the root, rounded to {0, 1}. `u[i, t+1]` is u_{i,t} (t in 0:T);
`v[i, t]`, `w[i, t]` are v_{i,t}, w_{i,t} (t in 1:T). `selected`/`selected_units` list the chosen
intervals (`IntervalIndex` ids) and their units (empty for 3-bin). `x` is the flat vector (u, v, w).
"""
struct Schedule
    u::Matrix{Float64}
    v::Matrix{Float64}
    w::Matrix{Float64}
    selected::Vector{Int}
    selected_units::Vector{Int}
    x::Vector{Float64}
end

function Schedule(u, v, w, selected, selected_units)
    return Schedule(
        u,
        v,
        w,
        selected,
        selected_units,
        vcat(vec(u), vec(v), vec(w)),
    )
end
