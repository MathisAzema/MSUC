# Experiences sur les instances SMS++ (Data/T-Ramp/*.nc4).
#
# Deux etapes, chacune parallelisee sur `Nbworkers` coeurs (un run par coeur) :
#
#   run_train     entraine la politique (SDDP, formulation 3-bin ou intervalle) et enregistre,
#                 pour chaque run, la borne inferieure, le temps passe dans chaque phase, ainsi que
#                 les plannings et les coupes qui permettent de rejouer la politique ;
#   run_simulate  rejoue la politique telle qu'elle etait a differents instants de
#                 l'entrainement et estime son cout par simulation.
#
# Utilisation, depuis la racine du depot (la premiere fois seulement :
# `julia --project=. -e 'import Pkg; Pkg.instantiate()'`), lancer `julia`, puis :
#
#     include("runSMS.jl")
#
#     size_list = [10, 50]                      # nombre de groupes de l'instance
#     id_list   = 1:5                           # identifiant de l'instance (Data/T-Ramp/<n>_0_<id>_w.nc4)
#     methods   = ["interval_dual", "bin3"]     # voir `MSUC.METHODS` ; "interval_projection", "interval_minimum" et
#                                               # "interval_maximum" : autres regles du point de linearisation (`interval_rule`)
#                                               # "bin3_gamma" : 3-bin avec les variables gamma en plus a la racine
#
#     # 1. Entrainement : un run de 3 600 s par combinaison (taille, identifiant, methode, S, sigma, ...).
#     run_train(size_list, id_list, methods, [3600]; S_list = [3, 5], sigma_list = [0.25, 1.0], mip_gap_list = [1e-2])
#
#     # 2. Simulation de la politique a 60, 600 et 3 600 s, sur 1 000 trajectoires. Les listes
#     #    doivent contenir les valeurs de l'entrainement (elles peuvent etre un sous-ensemble).
#     run_simulate(size_list, id_list, methods, [3600], [60, 600, 3600], 1000; S_list = [3, 5], sigma_list = [0.25, 1.0], mip_gap_list = [1e-2])
#
# Les tailles disponibles sont 10, 20, 50, 75, 100, 150 et 200 ; les identifiants vont de 1 a 5, sauf pour
# 200 groupes (1 a 11). Une combinaison sans fichier est ignoree avec un avertissement.
#
# On peut lancer `run_simulate` plus tard, dans une autre session, avec d'autres instants ou un
# autre nombre de trajectoires : tout ce qu'il faut est dans le dossier de chaque run.
#
# L'entrainement n'est pas modifie : `train` est appele tel quel. On releve seulement, apres chaque
# iteration, le planning x^k que le MIP racine a donne (les intervalles selectionnes) et les temps par
# phase ; le reste est fait apres l'entrainement.
#
# La politique de l'iteration k est celle que sa passe avant a simulee : le planning x^k, et les
# coupes d'etape qui existaient quand il a ete calcule, c'est-a-dire celles jusqu'a l'iteration k-1
# (la borne `bound` de l'iteration k repose sur ces memes coupes). Evaluer une coupe en x^k ne demande
# que ses coefficients sur les composantes que x^k selectionne : on ne garde donc, pour chaque coupe,
# que ses coefficients sur les composantes selectionnees par au moins un planning de l'entrainement
# (3 % des intervalles dans l'essai sur 10_0_1_w). La simulation est exacte et ne resout jamais le
# MIP racine.
#
# Options communes a `run_train` et `run_simulate` (valeurs par defaut entre parentheses). Les listes
# donnent un run par element de leur produit cartesien (avec les tailles, les identifiants, les methodes
# et les temps) :
#   T (24)                       horizon
#   S_list ([5])                 tailles du support de l'innovation ARMA
#   sigma_list ([1.0])           amplitude de l'alea : sigma multiplie l'ecart-type de l'innovation de
#                                l'ARMA (variance = 2.9e-4 * sigma^2) ; meme role que `force` de `PolicyGraph`
#                                (sigma = 0 : probleme deterministe)
#   mip_gap_list ([nothing])     ecarts d'optimalite du MIP racine (nothing : defaut du solveur)
#   passes_per_mip_list ([1])    passes avant/arriere entre deux resolutions du MIP racine
#
# La graine est fixee a 1 pour tous les runs (`MSUC.SEED`), entrainement et simulation.
#
# Fichiers ecrits dans results_sms/<methode>_T<T>_S<S>_sigma<s>_passes<p>_gap<g>/<instance>/time_<t>/ :
#
#   config.csv                   parametres du run, nombre d'iterations, temps total, derniere borne
#   log.csv                      une ligne par iteration : iteration, bound, incumbent, time, total_solves
#                                (`bound` = borne inferieure ; `incumbent` = valeur de l'incumbent du MIP
#                                racine, egale a `bound` si le MIP est resolu a l'optimalite et plus grande
#                                sinon ; `time` = secondes depuis le debut)
#   states.csv                   etats echantillonnes par les passes avant : une ligne par (iteration, passe,
#                                noeud), colonnes `power_<unite>` et `zeta_<i>` = etat (y, zeta) a la sortie du
#                                noeud ; le noeud 1 est l'etat initial. `passes_per_mip` passes par iteration
#   phases.csv                   une ligne par iteration : temps CUMULE (s) passe dans chaque phase,
#                                une colonne par phase. Les phases imbriquees portent le chemin de leur
#                                parent, par exemple `backward_pass/compute_coefficients`. Le temps d'une
#                                phase entre deux instants est la difference de deux lignes.
#   timers.csv                   bilan final : phase, nombre d'appels, temps (s)
#   schedules.csv               une ligne par iteration : intervalles selectionnes par le MIP racine
#                                (`selected`) et composantes non nulles de x = (u, v, w) (`x_nonzero`),
#                                listes separees par des points-virgules
#   used_components.csv          composantes de x utiles a au moins un planning (voir plus haut)
#   cuts.csv                     coupes d'etape restreintes a ces composantes, sans en-tete, une ligne par
#                                coupe, dans l'ordre de creation : noeud, alpha, delta, tau, beta
#   simulation_<N>.csv           resume de `run_simulate` : une ligne par instant (derniere iteration
#                                terminee a cet instant, sa borne, nombre de coupes d'etape, cout espere de
#                                la politique, cout d'engagement, erreur standard et intervalle de
#                                confiance a 95 % `ci95_low`, `ci95_high` : cout espere +- 1.96 erreur standard)
#   simulation_<t>_<N>.csv       cout de dispatch de chaque trajectoire a l'instant t
#
# Les phases de `phases.csv` sont : solve_first_stage_problem (MIP racine), set_schedule (planning
# fixe dans les etapes), forward_pass, backward_pass, et leurs sous-phases (solve_subproblem,
# get_dual_solution, compute_coefficients, add_cut, ...).
#
# Le travail d'un run (construire le modele, entrainer, ecrire et relire la politique, simuler) est dans
# `src/functions/run/` ; ce fichier ne contient que la mise en place de `Distributed` et les deux
# points d'entree, `run_train` et `run_simulate`, qui repartissent les runs sur les workers.

import Pkg
Pkg.activate(".")

using Distributed

# Nombre de runs menes en parallele, un coeur chacun (`NBWORKERS=2 julia` pour un essai). `GurobiThreads`
# est le nombre de fils de Gurobi pour chaque run. Garder Nbworkers * GurobiThreads <= nombre de coeurs
# physiques : le temps est la quantite mesuree, une machine sur-souscrite fausserait les resultats.
GurobiThreads = 1
Nbworkers = parse(Int, get(ENV, "NBWORKERS", "10"))

# `nprocs() > 1` n'est vrai que s'il y a de vrais workers (le master est compte dans `nworkers()`).
if nprocs() > 1
    rmprocs(workers())
end
addprocs(Nbworkers)

@everywhere import Pkg
@everywhere Pkg.activate(".")
@everywhere using MSUC
@everywhere using Gurobi

# Un environnement Gurobi par worker, enregistre dans le package : les jobs le relisent localement,
# `pmap` n'a rien a envoyer (sinon Distributed tenterait de copier l'environnement du master).
@everywhere const GRB_ENV = Gurobi.Env()
@everywhere Gurobi.GRBsetintparam(GRB_ENV, "Threads", $GurobiThreads)
@everywhere MSUC.set_run_optimizer!(() -> Gurobi.Optimizer(GRB_ENV))

"""
    run_train(size_list, id_list, methods, time_max_list; T, S_list, sigma_list, mip_gap_list,
              passes_per_mip_list)

Entraine un modele par combinaison (taille, identifiant, methode, temps limite, S, sigma, mip_gap,
passes_per_mip), `Nbworkers` a la fois, et enregistre les resultats de chaque run dans son dossier (voir
l'en-tete du fichier). `size_list` et `id_list` designent les instances de `Data/T-Ramp` (nombre de groupes
et identifiant), `methods` des cles de `MSUC.METHODS`.
"""
function run_train(
    size_list,
    id_list,
    method_list,
    time_max_list;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [nothing],
    passes_per_mip_list = [1],
)
    combos = MSUC.run_combos(
        size_list, id_list, method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
    pmap(combos; on_error = e -> (@warn("job d'entrainement en echec", exception = e); nothing)) do (instance, method, time_max, settings)
        MSUC.train_job(instance, method, time_max, settings)
        MSUC.release_memory()
    end
    return
end

"""
    run_simulate(size_list, id_list, methods, time_max_list, time_list, N; T, S_list, sigma_list,
                 mip_gap_list, passes_per_mip_list)

Simule la politique de chaque run entraine (memes arguments que `run_train`, memes options) a chacun
des instants de `time_list` (secondes depuis le debut de l'entrainement), sur `N` trajectoires. Si le
nombre de trajectoires `S^T` ne depasse pas 4096 elles sont toutes enumerees et le cout est exact.
Ecrit `simulation_<N>.csv` et `simulation_<instant>_<N>.csv` dans le dossier de chaque run.
"""
function run_simulate(
    size_list,
    id_list,
    method_list,
    time_max_list,
    time_list,
    N::Int;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [nothing],
    passes_per_mip_list = [1],
)
    combos = MSUC.run_combos(
        size_list, id_list, method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
    pmap(combos; on_error = e -> (@warn("job de simulation en echec", exception = e); nothing)) do (instance, method, time_max, settings)
        folder = MSUC.results_folder(instance, method, time_max, settings)
        MSUC.simulate_job(folder, instance, method, time_list, N, settings)
        MSUC.release_memory()
    end
    return
end
