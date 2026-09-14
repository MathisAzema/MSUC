# Exemple jouet : un horizon court ou les DEUX formulations convergent, ce qui permet
# de verifier numeriquement la Proposition 3 (elles ont le meme optimum) et de mesurer
# l'ecart de vitesse de convergence.
#
# A T = 24 la 3-bin est trop lente pour converger (elle plafonne vers 5 % de l'optimum) ;
# a T = 4 elle y arrive, et l'on peut comparer les deux sur le meme pied.
#
#   julia --project=. examples/toy_convergence.jl

using MSUC, Gurobi, Printf

const GRB_ENV = Gurobi.Env()
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

T        = 4            # horizon court : la 3-bin converge
ITER     = 25
FORCE    = 0.0          # deterministe => l'optimum exact est calculable
TOL      = 1e-6
INSTANCE = "Data/T-Ramp/10_0_1_w.nc4"

instance = MSUC.parse_nc4(INSTANCE, optimizer, T)
@printf("jouet : %s, T = %d, %d unites, %d intervalles\n\n",
        basename(INSTANCE), T, instance.N,
        sum(length(u.intervals) for u in instance.Thermalunits))

# Optimum exact du probleme deterministe.
opt = MSUC.bin_extensive_neutral(
    instance; silent = true, force = FORCE, S = 1, batch = 1,
    gap = 1e-9, timelimit = 120,
)[3]
@printf("optimum exact (formulation extensive) = %.2f\n\n", opt)

function converge(builder, method, label)
    model = MSUC.PolicyGraph(
        builder, instance, FORCE, MSUC.LinearGraph(T + 1), method;
        sense = :Min, lower_bound = 0.0, optimizer = optimizer,
    )
    MSUC.train(model; iteration_limit = ITER, print_level = 0)
    b = [l.bound for l in model.most_recent_training_results.log]
    k = findfirst(x -> (opt - x) / opt < TOL, b)
    @printf("%-22s borne = %.2f   gap = %.1e   converge a l'iteration %s\n",
            label, b[end], (opt - b[end]) / opt, k === nothing ? "> $ITER" : string(k))
    return b, k
end

b_ext, k_ext = converge(MSUC.subproblem_builder_UC_extended, MSUC.ExtendedFormulation,
                        "intervalle (39)")
b_3bin, k_3bin = converge(MSUC.subproblem_builder_UC_def, MSUC.SDDPFormulation,
                          "3-bin (13)")

println("\n it   intervalle (39)   3-bin (13)     % optimum (int. / 3-bin)")
for k in 1:min(ITER, length(b_ext), length(b_3bin))
    @printf("%3d  %15.2f  %13.2f     %6.2f / %6.2f\n",
            k, b_ext[k], b_3bin[k], 100 * b_ext[k] / opt, 100 * b_3bin[k] / opt)
end

# Proposition 3 : les deux formulations ont le meme optimum.
@assert abs(b_ext[end] - b_3bin[end]) / opt < TOL
@assert b_ext[end] <= opt * (1 + TOL) && b_3bin[end] <= opt * (1 + TOL)
@printf("\nOK : les deux bornes convergent vers le meme optimum (Prop. 3),")
@printf(" en %s et %s iterations.\n", k_ext, k_3bin)
