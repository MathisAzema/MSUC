"""
    solve_first_stage_problem(model, initial_state)

Resout le probleme racine (MIP), algorithme 2 lignes 4-5 : choisit le planning x^k (et gamma^k
en formulation intervalle) avec l'approximation courante du cout futur. Ne fixe pas x^k pour les
etapes de dispatch : c'est `one_iteration` qui le fait (`extract_schedule`, `set_schedule!`).

Renvoie la borne inferieure L^k, c'est-a-dire la borne DUALE du MIP (`objective_bound`) : avec un
`MIPGap` non nul, la valeur de l'incumbent majore l'optimum du MIP et n'est pas une borne
inferieure valide.
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
    return JuMP.objective_bound(root.subproblem)
end
