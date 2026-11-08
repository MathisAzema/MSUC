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

# Deux fonctions a passer a `train` et ce qu'elles remplissent. Aucune ne touche au deroulement de
# l'algorithme.
#   - `record_pass` (`forward_pass_callback`) : apres chaque passe avant, garde les etats echantillonnes
#     de la trajectoire, `(sampled_states_power, sampled_states_zeta)`, c'est-a-dire (y, zeta) a la
#     sortie de chaque noeud, le noeud 1 (racine) etant l'etat initial. Avec `interval_states = true`, garde
#     aussi `sampled_states_interval`, l'etat par intervalle ytilde^k de la passe avant (formulation
#     intervalle), avec les signaux de cout `sampled_signals_interval` (z^F) : un vecteur par intervalle et par
#     noeud, trop volumineux pour un long run, utile pour les diagnostics ;
#   - `record` (`post_iteration_callback`) : apres chaque iteration, releve les temps cumules par phase,
#     le planning x^k de l'iteration (celui du MIP racine) et les trajectoires de ses passes.
# `states[k][p]` est la trajectoire de la passe p de l'iteration k.
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
