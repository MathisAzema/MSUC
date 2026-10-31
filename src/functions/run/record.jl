# --- Enregistrement pendant l'entrainement ------------------------------------------------------

# Temps cumule (secondes) et nombre d'appels de chaque phase, cle = chemin de la phase
# (`parent/enfant`). Les temps sont cumules depuis le debut de `train`.
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

# Fonction a passer a `train(; post_iteration_callback)` : apres chaque iteration, releve les temps
# cumules par phase et le planning x^k de l'iteration (celui du MIP racine). Elle ne touche pas au
# deroulement de l'algorithme.
function make_recorder(model)
    snapshots = Dict{String,Tuple{Int,Float64}}[]
    schedules = Schedule[]
    function record(result)
        push!(snapshots, flatten_timer(model.timer_output))
        push!(schedules, model.schedule)
        return
    end
    return record, snapshots, schedules
end
