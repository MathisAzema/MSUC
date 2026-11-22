# Cumulative time (s) and call count of each phase, keyed by path (`parent/child`).
function flatten_timer(to::TimerOutputs.TimerOutput)
    phases = Dict{String,Tuple{Int,Float64}}()
    function walk(d, prefix)
        for (name, inner) in d["inner_timers"]
            path = isempty(prefix) ? name : prefix * "/" * name
            phases[path] = (inner["n_calls"], inner["time_ns"] / 1e9)
            walk(inner, path)
        end
    end
    walk(TimerOutputs.todict(to), "")
    return phases
end

# Returns callbacks for `train` that record without affecting the algorithm:
#   `record_pass` (`forward_pass_callback`) keeps the sampled (y, zeta) trajectory of each forward
#   pass; with `interval_states = true` it also keeps the per-interval states and cost signals
#   (large, for diagnostics only);
#   `record` (`post_iteration_callback`) keeps the phase timings, the schedule x^k and the pass
#   trajectories of the iteration.
# `states[k][p]` is the trajectory of pass p of iteration k.
function make_recorder(model; interval_states::Bool = false)
    snapshots = Dict{String,Tuple{Int,Float64}}[]
    schedules = Schedule[]
    states = Vector{NamedTuple}[]
    trajectories = NamedTuple[]
    function record_pass(trajectory)
        kept = interval_states ? trajectory : (
            sampled_states_power = trajectory.sampled_states_power,
            sampled_states_zeta = trajectory.sampled_states_zeta,
        )
        push!(trajectories, kept)
        return
    end
    function record(result)
        push!(snapshots, flatten_timer(model.timer_output))
        push!(schedules, model.schedule)
        push!(states, copy(trajectories))
        empty!(trajectories)
        return
    end
    return record, record_pass, snapshots, schedules, states
end
