# =============================================================================
#  Simulation de la politique induite par un modele entraine.
#
#  Donne un MAJORANT de l'optimum multi-etages : c'est le cout d'une politique
#  admissible precise. Avec la borne inferieure de l'entrainement, on encadre donc
#  l'optimum sans rien supposer :
#
#      borne inferieure  <=  optimum  <=  cout de la politique
#
#  Quand S^T est petit, tous les chemins sont enumeres et le cout espere est EXACT
#  (aucune erreur d'echantillonnage), ce qui en fait un test de validation net.
# =============================================================================

"""
    simulate_policy(model; max_paths, samples, seed) -> NamedTuple

Cout espere de la politique induite par `model` apres entrainement.

L'engagement est fixe une fois pour toutes : la racine est resolue, son etat sortant
(qui contient `x`) devient l'etat entrant commun a tous les chemins, et les coupes
intervalles sont reinstanciees a ce planning. Chaque chemin est ensuite parcouru en
resolvant les sous-problemes d'etage, comme le ferait l'operateur en temps reel.

Le cout d'etage est `stage_objective_value`, qui EXCLUT la variable d'epigraphe : on
mesure le cout reellement encouru, pas l'approximation de la valeur future.

Enumere les `S^T` chemins si ce nombre est <= `max_paths` (cout espere exact), sinon
tire `samples` chemins selon leur loi (estimateur sans biais, avec `seed`).

Renvoie `(cost, commitment_cost, dispatch_cost, exhaustive, npaths, costs, weights)`.
"""
function simulate_policy(
    model::PolicyGraph;
    max_paths::Int = 4096,
    samples::Int = 1000,
    seed::Int = 1234,
)
    T = model.instance.TimeHorizon
    S = length(model[2].noise_terms)

    # 1. Engagement : resoudre la racine, figer le planning, reinstancier les coupes.
    root = model[1]
    JuMP.optimize!(root.subproblem)
    commitment_cost = stage_objective_value(root, root.stage_objective)
    if model.method == ExtendedFormulation
        selected = Tuple{Int,Int,Int}[
            k for (k, v) in root.states_gamma if JuMP.value(v) > 0.5
        ]
        reinstantiate_interval_cuts!(model, selected)
    end
    root_state = get_outgoing_state(root)

    # 2. Chemins de l'arbre.
    exhaustive = float(S)^T <= max_paths
    paths, weights = if exhaustive
        ps = vec(collect(Iterators.product(ntuple(_ -> 1:S, T)...)))
        (
            [collect(p) for p in ps],
            [prod(model[1+t].noise_terms[p[t]].probability for t in 1:T) for p in ps],
        )
    else
        rng = Random.MersenneTwister(seed)
        probs = [n.probability for n in model[2].noise_terms]
        cum = cumsum(probs)
        draw() = [searchsortedfirst(cum, rand(rng)) for _ in 1:T]
        ([draw() for _ in 1:samples], fill(1 / samples, samples))
    end

    # 3. Parcours de chaque chemin avec la politique.
    costs = Float64[]
    for path in paths
        state = copy(root_state)
        c = 0.0
        for t in 1:T
            node = model[1+t]
            noise = node.noise_terms[path[t]].term
            res = solve_subproblem(
                model,
                node,
                state,
                noise,
                Tuple{Int,Any}[(1 + t, noise)];
                duality_handler = nothing,
            )
            c += res.stage_objective
            state = res.state
        end
        push!(costs, c)
    end
    dispatch_cost = sum(w * c for (w, c) in zip(weights, costs))
    return (
        cost = commitment_cost + dispatch_cost,
        commitment_cost = commitment_cost,
        dispatch_cost = dispatch_cost,
        exhaustive = exhaustive,
        npaths = length(paths),
        costs = costs,
        weights = weights,
    )
end
