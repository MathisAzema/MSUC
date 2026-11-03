#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function master_loop(model::PolicyGraph, options::Options)
    while true
        result = one_iteration(model, options)
        options.post_iteration_callback(result)
        log_iteration(options)
        if result.status != :not_solved
            return result.status
        end
    end
end

function termination_status(model::PolicyGraph)
    if model.most_recent_training_results === nothing
        return :model_not_solved
    end
    return model.most_recent_training_results.status
end

"""
    train(model; iteration_limit, time_limit, mip_gap, ...)

Entraine la politique (algorithme 2). Il faut au moins un critere d'arret : `iteration_limit`
(nombre d'iterations) et/ou `time_limit` (temps total, en secondes).

`passes_per_mip` (1 par defaut) est le nombre de passes avant/arriere faites, planning fixe, entre deux
resolutions du MIP racine : avec 1, c'est l'algorithme usuel ; avec 10, on affine les coupes par 10 passes
(des LP) avant de re-resoudre le MIP. Une « iteration » (`iteration_limit`, une ligne du journal) est un cycle
complet : une resolution du MIP racine suivie de `passes_per_mip` passes.

`mip_gap` fixe l'ecart d'optimalite relatif du MIP racine (par exemple `1e-2`) ; par defaut, celui du
solveur. Un ecart plus large accelere nettement l'entrainement (le MIP racine en est le poste
dominant) sans invalider la borne inferieure, qui reste la borne duale du MIP.

`linearization` (formulation intervalle seulement) choisit le point ytilde^k ou l'on linearise le probleme
local des intervalles non choisis par le planning (eq. (40)) : tout point atteignable donne une coupe valide,
serree au point visite pour les intervalles choisis, et ce choix ne change que la force de la coupe aux autres
plannings.
  * `:max_reachable` (defaut) : calcule a la passe arriere, la production echantillonnee si l'intervalle est
    choisi ou atteignable depuis un intervalle choisi, la production maximale atteignable sinon
    (`linearization_point`) ;
  * `:price` : regle des prix de la Sect. 4.2, calculee a la passe avant a partir des multiplicateurs du
    probleme d'etape echantillonne (`forward_interval_state`) : un intervalle non choisi monte a fond si son signal
    de cout est negatif et descend a fond sinon. La passe avant lit alors les duaux de chaque etape.
Sans effet en formulation 3-bin.

`reuse_incumbent` (faux par defaut) evite de resoudre le MIP racine quand c'est inutile. A chaque iteration,
le planning precedent est evalue avec la nouvelle approximation de la valeur future (cout d'engagement plus
maximum des coupes de la racine, nouvelle coupe comprise). S'il est a moins de l'ecart d'optimalite du MIP
(`mip_gap`, ou celui du solveur) de la meilleure borne inferieure, il est deja optimal pour le nouveau MIP
a cet ecart pres : on le garde, sans resoudre. La borne publiee reste alors celle de l'iteration precedente
et l'iteration est marquee `MIP evite` dans le journal (`Log.first_stage_solved`).
"""
function train(
    model::PolicyGraph;
    iteration_limit::Union{Int,Nothing} = nothing,
    time_limit::Union{Real,Nothing} = nothing,
    mip_gap::Union{Real,Nothing} = nothing,
    passes_per_mip::Int = 1,
    linearization::Symbol = :max_reachable,
    reuse_incumbent::Bool = false,
    print_level::Int = 1,
    log_frequency::Union{Nothing,Int} = nothing,
    log_every_seconds::Float64 = log_frequency === nothing ? -1.0 : 0.0,
    log_every_iteration::Bool = false,
    run_numerical_stability_report::Bool = true,
    add_to_existing_cuts::Bool = false,
    forward_pass_callback::Function = (x) -> nothing,
    post_iteration_callback = result -> nothing,
)
    if iteration_limit === nothing && time_limit === nothing
        throw(
            ArgumentError(
                "`train` needs a stopping criterion: pass `iteration_limit` and/or `time_limit`.",
            ),
        )
    end
    if passes_per_mip < 1
        throw(ArgumentError("`passes_per_mip` must be at least 1. Got $passes_per_mip."))
    end
    if !(linearization in (:max_reachable, :price))
        throw(
            ArgumentError(
                "`linearization` must be `:max_reachable` or `:price`. Got `:$linearization`.",
            ),
        )
    end
    model.ext[:linearization] = linearization
    model.ext[:reuse_incumbent] = reuse_incumbent
    if mip_gap !== nothing
        JuMP.set_optimizer_attribute(
            model[model.first_node].subproblem,
            MOI.RelativeGapTolerance(),
            Float64(mip_gap),
        )
    end
    log_frequency = something(log_frequency, 1)
    if log_frequency <= 0
        msg = "`log_frequency` must be at least `1`. Got $log_frequency."
        throw(ArgumentError(msg))
    end
    if log_every_iteration
        log_frequency = 1
        log_every_seconds = 0.0
    end
    function log_frequency_f(log::Vector{Log})
        if mod(length(log), log_frequency) != 0
            return false
        end
        last = options.last_log_iteration[]
        if last == 0
            return true
        elseif last == length(log)
            return false
        end
        seconds = log_every_seconds
        if log_every_seconds < 0.0
            if log[end].time <= 10
                seconds = 1.0
            elseif log[end].time <= 120
                seconds = 5.0
            else
                seconds = 30.0
            end
        end
        return log[end].time - log[last].time >= seconds
    end

    if !add_to_existing_cuts && model.most_recent_training_results !== nothing
        @warn("""
        Re-training a model with existing cuts!

        Are you sure you want to do this? The output from this training may be
        misleading because the policy is already partially trained.

        If you meant to train a new policy with different settings, you must
        build a new model.

        If you meant to refine a previously trained policy, turn off this
        warning by passing `add_to_existing_cuts = true` as a keyword argument
        to `MSUC.train`.

        In a future release, this warning may turn into an error.
        """)
    end
    TimerOutputs.reset_timer!(model.timer_output)
    log = Log[]

    if print_level > 0
        print_banner(stdout)
        print_problem_statistics(
            stdout,
            model,
            model.most_recent_training_results !== nothing,
        )
    end
    if run_numerical_stability_report
        report = sprint(
            io -> numerical_stability_report(
                io,
                model;
                print = print_level > 0,
            ),
        )
        print(report)
    end
    if print_level > 0
        print_iteration_header(stdout)
    end
    options = Options(
        model,
        initial_state(model.instance);
        iteration_limit,
        time_limit = time_limit === nothing ? nothing : Float64(time_limit),
        passes_per_mip,
        print_level,
        start_time = time(),
        log,
        log_frequency = log_frequency_f,
        forward_pass_callback,
        post_iteration_callback,
    )
    status = :not_solved
    try
        status = master_loop(model, options)
    catch ex
        if ex isa InterruptException
            status = :interrupted
        else
            throw(ex)
        end
    end
    training_results = TrainingResults(status, log)
    model.most_recent_training_results = training_results
    if print_level > 0
        log_iteration(options; force_if_needed = true)
        print_footer(stdout, training_results)
        if print_level > 1
            TimerOutputs.print_timer(stdout, model.timer_output)
            println()
        end
    end
    return
end
