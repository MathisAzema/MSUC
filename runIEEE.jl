# Experiences sur l'instance IEEE 118 noeuds (Data/IEEE/118_syst) : 54 groupes, reseau en flux DC.
#
# Meme fonctionnement que `runSMS.jl`, restreint a cette seule instance : deux etapes, chacune
# parallelisee sur `Nbworkers` coeurs (un run par coeur) :
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
#     include("runIEEE.jl")
#
#     methods = ["interval", "bin3"]            # voir `MSUC.METHODS`
#
#     # 1. Entrainement : un run de 3 600 s par combinaison (methode, S, sigma, ...).
#     run_train(methods, [3600]; S_list = [3, 5], sigma_list = [0.25, 1.0])
#
#     # 2. Simulation de la politique a 60, 600 et 3 600 s, sur 1 000 trajectoires. Les listes
#     #    doivent contenir les valeurs de l'entrainement (elles peuvent etre un sous-ensemble).
#     run_simulate(methods, [3600], [60, 600, 3600], 1000; S_list = [3, 5], sigma_list = [0.25, 1.0])
#
# On peut lancer `run_simulate` plus tard, dans une autre session, avec d'autres instants ou un
# autre nombre de trajectoires : tout ce qu'il faut est dans le dossier de chaque run.
#
# Options communes a `run_train` et `run_simulate` (valeurs par defaut entre parentheses). Les listes
# donnent un run par element de leur produit cartesien (avec les methodes et les temps) :
#   T (24)                       horizon (au plus 24 : le profil de charge couvre 24 heures)
#   S_list ([5])                 tailles du support de l'innovation ARMA
#   sigma_list ([1.0])           amplitude de l'alea : sigma multiplie l'ecart-type de l'innovation de
#                                l'ARMA (variance = 2.9e-4 * sigma^2) ; meme role que `force` de `PolicyGraph`
#                                (sigma = 0 : probleme deterministe)
#   mip_gap_list ([1e-2])        ecarts d'optimalite du MIP racine (nothing : defaut du solveur, 1e-4) ;
#                                le MIP racine est le poste dominant du temps d'entrainement
#   passes_per_mip_list ([1])    passes avant/arriere entre deux resolutions du MIP racine
#
# La graine est fixee a 1 pour tous les runs (`MSUC.SEED`), entrainement et simulation.
#
# Les fichiers ecrits sont ceux de `runSMS.jl` (config.csv, log.csv, states.csv, phases.csv, timers.csv,
# schedules.csv, used_components.csv, cuts.csv, simulation_<N>.csv, simulation_<t>_<N>.csv), dans
#
#     results_ieee/<methode>_T<T>_S<S>_sigma<s>_passes<p>_gap<g>/118_syst/time_<t>/
#
# Le travail d'un run (construire le modele, entrainer, ecrire et relire la politique, simuler) est dans
# `src/functions/run/`, partage avec `runSMS.jl` ; ce fichier ne contient que la mise en place de
# `Distributed` et les deux points d'entree, qui repartissent les runs sur les workers.

import Pkg
Pkg.activate(".")

using Distributed

# Nombre de runs menes en parallele, un coeur chacun (`NBWORKERS=2 julia` pour changer). Une seule instance :
# il y a au plus un run par methode et par reglage, donc 2 workers suffisent pour comparer les deux methodes.
# `GurobiThreads` est le nombre de fils de Gurobi pour chaque run. Garder Nbworkers * GurobiThreads <= nombre
# de coeurs physiques : le temps est la quantite mesuree, une machine sur-souscrite fausserait les resultats.
GurobiThreads = 1
Nbworkers = parse(Int, get(ENV, "NBWORKERS", "2"))

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
    run_train(methods, time_max_list; T, S_list, sigma_list, mip_gap_list, passes_per_mip_list)

Entraine un modele par combinaison (methode, temps limite, S, sigma, mip_gap, passes_per_mip) sur l'instance
IEEE 118 noeuds, `Nbworkers` a la fois, et enregistre les resultats de chaque run dans son dossier (voir
l'en-tete du fichier). `methods` designe des cles de `MSUC.METHODS`.
"""
function run_train(
    method_list,
    time_max_list;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [1e-2],
    passes_per_mip_list = [1],
)
    combos = MSUC.ieee_combos(
        method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
    pmap(combos; on_error = e -> (@warn("job d'entrainement en echec", exception = e); nothing)) do (instance, method, time_max, settings)
        MSUC.train_job(instance, method, time_max, settings)
        MSUC.release_memory()
    end
    return
end

"""
    run_simulate(methods, time_max_list, time_list, N; T, S_list, sigma_list, mip_gap_list,
                 passes_per_mip_list)

Simule la politique de chaque run entraine (memes arguments que `run_train`, memes options) a chacun
des instants de `time_list` (secondes depuis le debut de l'entrainement), sur `N` trajectoires. Si le
nombre de trajectoires `S^T` ne depasse pas 4096 elles sont toutes enumerees et le cout est exact.
Ecrit `simulation_<N>.csv` et `simulation_<instant>_<N>.csv` dans le dossier de chaque run.
"""
function run_simulate(
    method_list,
    time_max_list,
    time_list,
    N::Int;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [1e-2],
    passes_per_mip_list = [1],
)
    combos = MSUC.ieee_combos(
        method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
    pmap(combos; on_error = e -> (@warn("job de simulation en echec", exception = e); nothing)) do (instance, method, time_max, settings)
        folder = MSUC.results_folder(instance, method, time_max, settings)
        MSUC.simulate_job(folder, instance, method, time_list, N, settings)
        MSUC.release_memory()
    end
    return
end
