# =============================================================================
#  Equivalent deterministe de l'arbre de scenarios : l'optimum EXACT du probleme
#  multi-etapes (1), non-anticipativite (1j) incluse.
#
#  Sert de reference aux bornes SDDP dans le regime stochastique. L'arbre a
#  sum_{t=0}^{T} S^t noeuds, donc n'est tractable que pour S et T petits : avec
#  S = 2 et T = 4, 31 noeuds.
#
#  A ne pas confondre avec `bin_extensive_neutral`, qui est DEUX-ETAGES : son
#  dispatch voit toute la trajectoire d'un coup, elle MINORE donc le multi-etapes.
# =============================================================================

"""
    scenario_tree(instance, T) -> (parent, innovation, probability, eta)

Arbre de scenarios de l'ARMA de l'instance, indexe par `(t, k)` avec `k in 1:S^t`.

Renvoie, pour chaque etage `t in 0:T`, le noeud parent, l'indice d'innovation, la
probabilite du noeud, et l'erreur relative `eta` de chaque noeud (vecteur de longueur
|N|). `eta` est deterministe le long d'un chemin : la recursion (1g) est donc evaluee
numeriquement, sans variable `zeta` dans le programme.
"""
function scenario_tree(instance::Instance, T::Int)
    a = instance.arma
    S = length(a.support)
    nb = size(a.S_eta, 1)
    parent = [zeros(Int, S^t) for t in 0:T]
    innovation = [zeros(Int, S^t) for t in 0:T]
    probability = [zeros(S^t) for t in 0:T]
    zeta = [[zeros(zeta_dimension(a)) for _ in 1:S^t] for t in 0:T]
    eta = [[zeros(nb) for _ in 1:S^t] for t in 0:T]

    probability[1][1] = 1.0
    zeta[1][1] = copy(a.zeta0)
    eta[1][1] = a.S_eta * a.zeta0
    for t in 1:T
        for k in 1:S^t
            p = div(k - 1, S) + 1            # parent a l'etage t-1
            s = mod(k - 1, S) + 1            # innovation tiree
            parent[t+1][k] = p
            innovation[t+1][k] = s
            probability[t+1][k] = probability[t][p] * a.probability[s]
            zeta[t+1][k] = a.Psi * zeta[t][p] + a.E * a.support[s]
            eta[t+1][k] = a.S_eta * zeta[t+1][k]
        end
    end
    return parent, innovation, probability, eta
end

"""
    multistage_extensive(instance; force, silent, gap, timelimit, res_gamma)

Resout l'equivalent deterministe de l'arbre de scenarios : l'optimum EXACT du probleme
multi-etapes (1). Les deux bornes SDDP doivent converger vers cette valeur.

L'engagement `x` est commun a tout l'arbre (decide en t = 0), le dispatch est indexe par
noeud de l'arbre, ce qui encode exactement la non-anticipativite (1j).

Les lignes de faisabilite d'arret (44) ne sont PAS imposees : sur l'horizon complet elles
sont impliquees par les rampes, elles ne servent qu'a la decomposition par etage. Leur
absence rend cette reference independante de (44) et (46).

Renvoie `(valeur, borne, temps, planning, gamma_par_unite)` ou `nothing` si echec.
`res_gamma` permet de figer l'engagement a un planning donne, pour evaluer un planning
issu du SDDP.
"""
function multistage_extensive(
    instance::Instance;
    force::Float64 = 1.0,
    silent::Bool = true,
    gap::Float64 = 1e-9,
    timelimit::Int = 300,
    res_gamma = nothing,
)
    T = instance.TimeHorizon
    units = instance.Thermalunits
    names = [u.name for u in units]
    Next = instance.Next
    Buses = 1:size(Next)[1]
    Lines = values(instance.Lines)
    S = length(instance.arma.support)
    _, _, probability, eta = scenario_tree(instance, T)
    nodes_at(t) = 1:S^t
    parent_of(t, k) = div(k - 1, S) + 1

    model = Model(instance.optimizer)
    silent && set_silent(model)

    # Engagement : premier etage, commun a tout l'arbre.
    @variable(model, is_on[j in names, t in 0:T], Bin)
    @variable(model, start_up[j in names, t in 1:T], Bin)
    @variable(model, start_down[j in names, t in 1:T], Bin)
    @variable(model, thermal_fixed_cost >= 0)
    thermal_unit_commit_constraints(model, instance)
    if res_gamma !== nothing
        @constraint(model, [u in units, t in 1:T],
            sum(res_gamma[u.name, a, b] for (a, b) in u.intervals if a <= t < b) == is_on[u.name, t])
        @constraint(model, [u in units, t in 1:T],
            sum(res_gamma[u.name, a, b] for (a, b) in u.intervals if a == t) == start_up[u.name, t])
        @constraint(model, [u in units, t in 1:T],
            sum(res_gamma[u.name, a, b] for (a, b) in u.intervals if b == t) == start_down[u.name, t])
    end

    # Dispatch : un jeu de variables par noeud de l'arbre (non-anticipativite 1j).
    power = Dict((j, t, k) => @variable(model, lower_bound = 0.0)
                 for j in names for t in 0:T for k in nodes_at(t))
    shed = Dict((b, t, k) => @variable(model, lower_bound = 0.0)
                for b in Buses for t in 1:T for k in nodes_at(t))
    curt = Dict((b, t, k) => @variable(model, lower_bound = 0.0)
                for b in Buses for t in 1:T for k in nodes_at(t))
    θ = Dict((b, t, k) => @variable(model) for b in Buses for t in 1:T for k in nodes_at(t))
    flow = Dict((b, bp, t, k) => @variable(model)
                for b in Buses for bp in Next[b] for t in 1:T for k in nodes_at(t))

    for u in units
        @constraint(model, power[(u.name, 0, 1)] == u.InitialPower)
    end
    for t in 1:T, k in nodes_at(t)
        kp = parent_of(t, k)
        for u in units
            j = u.name
            # (43a) capacite
            @constraint(model, power[(j, t, k)] <= u.MaxPower * is_on[j, t])
            @constraint(model, power[(j, t, k)] >= u.MinPower * is_on[j, t])
            # (43b)-(43c) rampes, S^U = S^D = MinPower
            @constraint(model,
                power[(j, t, k)] - power[(j, t - 1, kp)] <=
                -u.DeltaRampUp * start_up[j, t] +
                (u.MinPower + u.DeltaRampUp) * is_on[j, t] - u.MinPower * is_on[j, t-1])
            @constraint(model,
                power[(j, t - 1, kp)] - power[(j, t, k)] <=
                -u.DeltaRampDown * start_down[j, t] +
                (u.MinPower + u.DeltaRampDown) * is_on[j, t-1] - u.MinPower * is_on[j, t])
        end
        # (1e)-(1f) reseau
        for line in Lines
            @constraint(model, flow[(line.b1, line.b2, t, k)] <= line.Fmax)
            @constraint(model, flow[(line.b1, line.b2, t, k)] >= -line.Fmax)
            @constraint(model, flow[(line.b1, line.b2, t, k)] ==
                               line.B12 * (θ[(line.b1, t, k)] - θ[(line.b2, t, k)]))
        end
        # (1h) bilan, avec l'erreur ARMA du noeud
        for b in Buses
            @constraint(model,
                sum(power[(u.name, t, k)] for u in units if u.Bus == b) - curt[(b, t, k)] +
                shed[(b, t, k)] ==
                instance.Demandbus[b][t] * (1 + force * eta[t+1][k][b]) +
                sum(flow[(b, bp, t, k)] for bp in Next[b]))
        end
    end

    @objective(model, Min,
        thermal_fixed_cost + sum(
            probability[t+1][k] * (
                sum(u.LinearTerm * power[(u.name, t, k)] for u in units) + sum(
                    SHEDDING_COST * shed[(b, t, k)] + CURTAILEMENT_COST * curt[(b, t, k)]
                    for b in Buses)
            ) for t in 1:T for k in nodes_at(t)))

    set_optimizer_attribute(model, "TimeLimit", timelimit)
    set_optimizer_attribute(model, "MIPGap", gap)
    start = time()
    optimize!(model)
    elapsed = time() - start
    primal_status(model) == MOI.FEASIBLE_POINT || return nothing
    schedule = [value.(is_on), value.(start_up), value.(start_down)]
    return objective_value(model), objective_bound(model), elapsed, schedule
end
