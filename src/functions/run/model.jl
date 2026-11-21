# --- Modele et dossier de resultats d'un run (voir `runSMS.jl` et `runIEEE.jl`) ----------------

# Les methodes d'un run : (constructeur des sous-problemes, formulation, regle `interval_rule` de `train`).
# "bin3" est la formulation 3-bin, "bin3_gamma" la meme avec les variables gamma en plus a la racine (coupes 3-bin
# inchangees ; a la racine, relaxation identique a la 3-bin) ; "interval_<regle>" la formulation intervalle avec la regle `<regle>` de construction
# du point de linearisation (`INTERVAL_RULES`), "interval_dual" etant la regle des prix de l'article. Le nom de la
# methode figure dans le dossier de resultats, donc les regles ne s'ecrasent pas.
const METHODS = Dict(
    "interval_dual" => (subproblem_builder_UC_extended, ExtendedFormulation, :dual),
    "interval_projection" => (subproblem_builder_UC_extended, ExtendedFormulation, :projection),
    "interval_minimum" => (subproblem_builder_UC_extended, ExtendedFormulation, :minimum),
    "interval_maximum" => (subproblem_builder_UC_extended, ExtendedFormulation, :maximum),
    "bin3" => (subproblem_builder_UC_def, SDDPFormulation, :dual),
    "bin3_gamma" => (subproblem_builder_UC_def_gamma, SDDPFormulation, :dual),
)

# Graine de tous les runs : entrainement (tirages de la passe avant) et simulation.
const SEED = 1

# Instance IEEE 118 noeuds : un dossier de `Data/IEEE` (lu par `parse_IEEE`), a la place d'un fichier
# `.nc4` de `Data/T-Ramp`. Les deux sortes d'instances se distinguent par leur nom : c'est un
# dossier de `Data/IEEE` ou non.
const IEEE_INSTANCE = "118_syst"

is_ieee(instance_name) =
    isdir(joinpath(@__DIR__, "..", "..", "..", "Data", "IEEE", instance_name))

# `settings` : (T, S, sigma, mip_gap, passes_per_mip). Tous les parametres figurent dans le nom du
# dossier, pour que des runs de parametres differents ne s'ecrasent jamais. Les runs IEEE sont ecrits
# dans `results_ieee/`, les runs SMS++ dans `results_sms/`.
function results_folder(instance_name, method, time_max, settings)
    gap = settings.mip_gap === nothing ? "default" : string(settings.mip_gap)
    tag = string(
        method, "_T", settings.T, "_S", settings.S, "_sigma", settings.sigma,
        "_passes", settings.passes_per_mip, "_gap", gap,
    )
    root = is_ieee(instance_name) ? "results_ieee" : "results_sms"
    return "$(root)/$(tag)/$(instance_name)/time_$(time_max)"
end

# Nom, sans extension, du fichier de `Data/T-Ramp` d'une instance : nombre de groupes, 0, identifiant, w.
instance_name(n_units, id) = "$(n_units)_0_$(id)_w"

# Tous les runs d'une campagne : le produit des listes. Chaque element est
# `(instance, methode, temps limite, settings)`. Les combinaisons (taille, identifiant) sans fichier
# dans `Data/T-Ramp` sont ignorees avec un avertissement (les instances a 200 groupes vont jusqu'a
# l'identifiant 11, les autres jusqu'a 5) ; il y a une erreur si aucune n'existe.
function run_combos(
    size_list,
    id_list,
    method_list,
    time_max_list;
    T,
    S_list,
    sigma_list,
    passes_per_mip_list,
    mip_gap_list,
)
    wanted = [instance_name(n_units, id) for n_units in size_list for id in id_list]
    instances = filter(name -> isfile("Data/T-Ramp/$(name).nc4"), wanted)
    absent = setdiff(wanted, instances)
    isempty(absent) || @warn "instances absentes de Data/T-Ramp, ignorees" absent
    isempty(instances) &&
        error("aucune instance de Data/T-Ramp pour size_list = $(size_list) et id_list = $(id_list)")
    return campaign(
        instances, method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
end

# Tous les runs de l'instance IEEE (`IEEE_INSTANCE`) : memes elements que `run_combos`.
ieee_combos(method_list, time_max_list; kwargs...) =
    campaign([IEEE_INSTANCE], method_list, time_max_list; kwargs...)

# Le produit des listes : un run `(instance, methode, temps limite, settings)` par combinaison.
function campaign(
    instances,
    method_list,
    time_max_list;
    T,
    S_list,
    sigma_list,
    passes_per_mip_list,
    mip_gap_list,
)
    return [
        (instance, method, time_max, (; T, S, sigma, mip_gap, passes_per_mip)) for
        instance in instances for method in method_list for time_max in time_max_list for
        S in S_list for sigma in sigma_list for passes_per_mip in passes_per_mip_list for
        mip_gap in mip_gap_list
    ]
end

# Fabrique d'optimiseur d'un run, enregistree dans chaque processus (`set_run_optimizer!`, appelee sur
# chaque worker par `runSMS.jl`). Les jobs la relisent ici au lieu de la recevoir de `pmap`, qui
# enverrait sinon l'environnement Gurobi du processus principal aux workers.
const RUN_OPTIMIZER = Ref{Any}(nothing)

set_run_optimizer!(factory) = (RUN_OPTIMIZER[] = factory; nothing)

function run_optimizer()
    RUN_OPTIMIZER[] === nothing && error(
        "aucun optimiseur enregistre : appeler `MSUC.set_run_optimizer!(() -> Gurobi.Optimizer(env))` " *
        "sur chaque processus (voir runSMS.jl)",
    )
    return RUN_OPTIMIZER[]
end

# Un modele neuf (sans coupe) : meme construction a l'entrainement et a la simulation.
function build_model(instance_name, method, settings, optimizer)
    builder, formulation, _ = METHODS[method]
    instance = if is_ieee(instance_name)
        parse_IEEE(instance_name, optimizer, settings.T; sigma = settings.sigma, S = settings.S)
    else
        parse_nc4(
            "Data/T-Ramp/$(instance_name).nc4",
            optimizer,
            settings.T;
            sigma = settings.sigma,
            S = settings.S,
        )
    end
    # force = 1 : l'amplitude de l'alea est entierement donnee par `sigma`.
    return PolicyGraph(
        builder,
        instance,
        1.0,
        Graph(settings.T + 1),
        formulation;
        lower_bound = 0.0,
        optimizer = optimizer,
    )
end
