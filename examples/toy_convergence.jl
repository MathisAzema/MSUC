# Exemple jouet : horizon court et support reduit, pour que TOUT soit verifiable.
#
# A T = 24 la 3-bin ne converge pas et l'arbre de scenarios est hors de portee. A
# T = 4 avec S = 2, l'arbre n'a que 2^4 = 16 feuilles : on calcule l'optimum
# multi-etages EXACT par son equivalent deterministe, et on encadre sans rien supposer
#
#     borne inferieure (entrainement)  <=  optimum  <=  cout de la politique (simulation)
#
# pour les deux formulations. C'est la verification la plus complete du depot : elle
# couvre la validite des coupes, l'equivalence de la Prop. 3, le processus ARMA, l'etat
# zeta et les coefficients delta.
#
#   julia --project=. examples/toy_convergence.jl

using MSUC, Gurobi, Printf, Random

const GRB_ENV = Gurobi.Env()
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

T     = 4       # horizon court : la 3-bin converge
S     = 2       # support de l'innovation : 2^4 = 16 scenarios, arbre tractable
FORCE = 1.0     # amplitude de l'alea ARMA
ITER  = 30
TOL   = 1e-6
SEED  = 1234    # la passe avant echantillonne via le rand() global : graine fixee
                # pour que l'exemple soit reproductible et que les deux
                # formulations voient la meme suite de scenarios

instance = MSUC.parse_nc4("Data/T-Ramp/10_0_1_w.nc4", optimizer, T; S = S)
@printf("jouet : T = %d, S = %d, %d unites, %d intervalles, %d scenarios\n\n",
        T, S, instance.N, sum(length(u.intervals) for u in instance.Thermalunits), S^T)

# --- Optimum exact du probleme multi-etages, par l'arbre de scenarios --------------
opt = MSUC.multistage_extensive(instance; force = FORCE, gap = 1e-9, timelimit = 300)[1]
@printf("optimum multi-etages EXACT (arbre de %d noeuds) = %.4f\n\n",
        sum(S^t for t in 0:T), opt)

function run(builder, method, label)
    model = MSUC.PolicyGraph(
        builder, instance, FORCE, MSUC.Graph(T + 1), method;
        lower_bound = 0.0, optimizer = optimizer,
    )
    Random.seed!(SEED)
    MSUC.train(model; iteration_limit = ITER, print_level = 0)
    bounds = [l.bound for l in model.most_recent_training_results.log]
    k = findfirst(x -> (opt - x) / opt < TOL, bounds)
    sim = MSUC.simulate(model)

    @printf("%-16s borne = %.4f   politique = %.4f   converge a l'iteration %s\n",
            label, bounds[end], sim.cost, k === nothing ? "> $ITER" : string(k))
    @printf("%-16s encadrement = %.2e   (%d chemins, exhaustif = %s)\n",
            "", (sim.cost - bounds[end]) / opt, sim.npaths, sim.exhaustive)

    # La borne minore l'optimum, le cout de la politique le majore.
    @assert bounds[end] <= opt * (1 + TOL)  "borne superieure a l'optimum"
    @assert sim.cost >= opt * (1 - TOL)     "politique inferieure a l'optimum"
    return bounds, sim
end

b_ext, s_ext = run(MSUC.subproblem_builder_UC_extended, MSUC.ExtendedFormulation, "intervalle (39)")
b_3bin, s_3bin = run(MSUC.subproblem_builder_UC_def, MSUC.SDDPFormulation, "3-bin (13)")

println("\n it   intervalle (39)   3-bin (13)     % optimum (int. / 3-bin)")
for k in 1:min(ITER, length(b_ext), length(b_3bin))
    @printf("%3d  %15.2f  %13.2f     %6.2f / %6.2f\n",
            k, b_ext[k], b_3bin[k], 100 * b_ext[k] / opt, 100 * b_3bin[k] / opt)
end

# Proposition 3 : les deux formulations ont le meme optimum.
@assert abs(b_ext[end] - b_3bin[end]) / opt < TOL
println()
@printf("OK  borne <= optimum <= politique, verifie pour les deux formulations.\n")
@printf("OK  les deux bornes convergent vers le meme optimum (Prop. 3).\n")
@printf("    optimum exact    %.4f\n", opt)
@printf("    intervalle       borne %.4f / politique %.4f\n", b_ext[end], s_ext.cost)
@printf("    3-bin            borne %.4f / politique %.4f\n", b_3bin[end], s_3bin.cost)
