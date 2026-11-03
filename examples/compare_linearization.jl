# Compare les deux regles de point de linearisation des coupes intervalle (`train(...; linearization)`) :
#
#   :max_reachable   regle par defaut, calculee a la passe arriere : la production echantillonnee si
#                    l'intervalle est choisi ou atteignable depuis un intervalle choisi, sa production
#                    maximale atteignable sinon ;
#   :price           regle des prix (Sect. 4.2, eq. (55)-(56)), calculee a la passe avant avec les duaux de
#                    chaque etape : un intervalle non choisi monte a fond si son signal de cout est negatif,
#                    descend a fond sinon.
#
# Les deux regles donnent des coupes valides, serrees au point visite : seule leur force aux autres
# plannings differe. Chaque graine fixe la suite de scenarios tires par la passe avant, de sorte que les
# deux regles voient les memes tirages (leurs trajectoires divergent ensuite, les coupes n'etant pas les memes).
#
#   julia --project=. examples/compare_linearization.jl

using MSUC, Gurobi, Printf, Random, Statistics

const GRB_ENV = Gurobi.Env(output_flag = 0)
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

INSTANCE = "ieee"     # "ieee" (Data/IEEE/118_syst) ou le nom d'un fichier de Data/T-Ramp, par exemple "10_0_1_w"
T        = 4          # horizon
S        = 2          # support de l'innovation ARMA
SIGMA    = 1.0        # amplitude de l'alea (0.0 : deterministe)
ITER     = 12         # iterations SDDP
MIP_GAP  = 1e-3       # ecart d'optimalite du MIP racine (nothing : defaut du solveur)
SEEDS    = 1:5        # graines de la passe avant
OPT      = 146328.4058   # optimum exact de l'arbre de scenarios (README, IEEE T = 4, S = 2) ; nothing : inconnu
RULES    = (:max_reachable, :price)

instance = INSTANCE == "ieee" ? MSUC.parse_IEEE("118_syst", optimizer, T; S = S, sigma = SIGMA) :
           MSUC.parse_nc4("Data/T-Ramp/$(INSTANCE).nc4", optimizer, T; S = S, sigma = SIGMA)
@printf("instance %s : %d unites, %d intervalles, T = %d, S = %d, sigma = %.2f, %d iterations, graines %s\n\n",
        INSTANCE, instance.N, sum(length(u.intervals) for u in instance.Thermalunits), T, S, SIGMA, ITER, SEEDS)

# entraine un modele neuf avec la regle donnee ; renvoie les bornes et les temps de passe avant / arriere
function train_rule(rule, seed, iterations)
    model = MSUC.PolicyGraph(
        MSUC.subproblem_builder_UC_extended, instance, 1.0, MSUC.Graph(T + 1), MSUC.ExtendedFormulation;
        lower_bound = 0.0, optimizer = optimizer,
    )
    Random.seed!(seed)
    MSUC.train(model; iteration_limit = iterations, mip_gap = MIP_GAP, print_level = 0,
               run_numerical_stability_report = false, linearization = rule)
    phases = MSUC.TimerOutputs.todict(model.timer_output)["inner_timers"]
    return (
        bounds = [l.bound for l in model.most_recent_training_results.log],
        forward = phases["forward_pass"]["time_ns"] / 1e9,
        backward = phases["backward_pass"]["time_ns"] / 1e9,
    )
end

# prechauffage : la premiere execution de chaque regle compile le code et fausserait les temps
for rule in RULES
    train_rule(rule, 1, 2)
end

results = Dict(rule => [train_rule(rule, seed, ITER) for seed in SEEDS] for rule in RULES)

pct(b) = OPT === nothing ? "" : @sprintf(" (%6.2f %%)", 100 * b / OPT)
println(" k   ", join([@sprintf("%-26s", string(rule, " (moyenne)")) for rule in RULES]), "  price >= max_reachable")
for k in 1:ITER
    means = [mean(r.bounds[k] for r in results[rule]) for rule in RULES]
    wins = count(i -> results[:price][i].bounds[k] >= results[:max_reachable][i].bounds[k] - 1e-6, eachindex(SEEDS))
    @printf("%2d   %s  %d / %d graines\n", k, join([@sprintf("%-26s", @sprintf("%.2f", m) * pct(m)) for m in means]), wins, length(SEEDS))
end

println("\nborne finale par graine :")
for (i, seed) in enumerate(SEEDS)
    @printf("  graine %d : %s\n", seed, join([@sprintf("%s %.2f%s", rule, results[rule][i].bounds[end], pct(results[rule][i].bounds[end])) for rule in RULES], "   "))
end
println("\ntemps cumule sur les graines (hors compilation) :")
for rule in RULES
    @printf("  %-14s passe avant %.2f s, passe arriere %.2f s\n", rule, sum(r.forward for r in results[rule]), sum(r.backward for r in results[rule]))
end

# Une borne inferieure valide ne peut pas depasser l'optimum exact : c'est le test.
if OPT !== nothing
    @assert all(r.bounds[end] <= OPT * (1 + 1e-6) for rule in RULES for r in results[rule])
    println("\nOK : toutes les bornes restent sous l'optimum exact.")
end
