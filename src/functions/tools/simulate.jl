#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    simulate(model; samples = 1000, max_paths = 4096, seed = 1234)

Simule la politique apres `train` : resout le probleme racine avec les coupes finales, fixe le
planning x^k obtenu (`set_schedule!`), puis resout les etapes de dispatch le long de trajectoires
d'alea. Quand le nombre de trajectoires `S^T` ne depasse pas `max_paths`, elles sont toutes
enumerees et le cout espere est EXACT ; sinon on tire `samples` trajectoires avec la graine `seed`.

Renvoie `(cost, commitment_cost, dispatch_cost, exhaustive, npaths, costs, weights)`. `cost` est le
cout espere de la politique, donc un majorant de l'optimum (valeur exacte si `exhaustive`).
"""
function simulate(
    model::PolicyGraph;
    samples::Int = 1000,
    max_paths::Int = 4096,
    seed::Int = 1234,
)
    T = model.instance.TimeHorizon
    S = length(model[2].noise_terms)

    # Probleme racine : planning de la politique et son cout d'engagement.
    root = model[model.first_node]
    state = initial_state(model.instance)
    solve_first_stage_problem(model, state)
    commitment_cost = stage_objective_value(root, root.stage_objective)
    set_schedule!(model, extract_schedule(model))

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

    # Etapes de dispatch le long de chaque trajectoire.
    costs = Float64[]
    for path in paths
        current = state
        c = 0.0
        for t in 1:T
            node = model[1+t]
            noise = node.noise_terms[path[t]].term
            res = solve_subproblem(model, node, current, noise)
            c += res.stage_objective
            current = res.state
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
