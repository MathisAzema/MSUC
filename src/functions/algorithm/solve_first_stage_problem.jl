"""
    solve_first_stage_problem(model, initial_state)

Resout le probleme racine (MIP), algorithme 2 lignes 4-5 : choisit le planning x^k (et gamma^k
en formulation intervalle) avec l'approximation courante du cout futur. Ne fixe pas x^k pour les
etapes de dispatch : c'est `one_iteration` qui le fait (`extract_schedule`, `set_schedule!`).

Renvoie `(bound, incumbent)`. `bound` est la borne inferieure L^k, c'est-a-dire la borne DUALE du
MIP (`objective_bound`) : avec un `MIPGap` non nul, la valeur de l'incumbent majore l'optimum du MIP
et n'est pas une borne inferieure valide. `incumbent` est la valeur de l'objectif du planning x^k
retenu (`objective_value`), egale a `bound` si le MIP est resolu a l'optimalite.
"""
function solve_first_stage_problem(model::PolicyGraph, initial_state)
    root = model[model.first_node]
    noise = root.noise_terms[1].term
    solve_subproblem(
        model,
        root,
        initial_state,
        noise,
    )
    return (
        bound = JuMP.objective_bound(root.subproblem),
        incumbent = JuMP.objective_value(root.subproblem),
    )
end

"""
    first_stage_value(model, schedule)

Valeur du planning `schedule` pour le probleme racine avec l'approximation courante : cout d'engagement plus
le maximum des coupes de la racine (coupe triviale comprise), c'est-a-dire l'objectif du MIP racine en ce
planning, sans le resoudre. Appelee apres une passe arriere, elle inclut la coupe qui vient d'y etre ajoutee.
"""
function first_stage_value(model::PolicyGraph, schedule::Schedule)
    root = model[model.first_node]
    return schedule_commitment_cost(model.instance, schedule) +
           maximum(root_cut_value(model, cut, schedule) for cut in root.cuts)
end

"""
    reusable_incumbent(model)

Decide si le MIP racine peut etre evite a cette iteration. Le planning x^k de l'iteration precedente est
evalue avec la nouvelle approximation (`first_stage_value`). Cette approximation ne decroit jamais, donc la
meilleure borne inferieure deja obtenue minore aussi le nouveau MIP racine ; si x^k est a moins de l'ecart
d'optimalite du MIP de cette borne (meme definition que Gurobi : `(valeur - borne) / |valeur|`), x^k est
deja optimal pour le nouveau MIP a cet ecart pres. Renvoie alors `(bound, incumbent)` avec la meilleure borne
(inchangee) et la valeur de x^k, sinon `nothing`. Il faut un planning deja choisi et une borne finie, donc
la premiere iteration resout toujours le MIP.
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
