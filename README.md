# MSUC.jl — SDDP à coupes intervalles pour l'unit commitment stochastique multi-étapes

Code de l'article **« Stronger cuts for stochastic dual dynamic programming for multistage
stochastic unit commitment problems based on interval variables »**
(M. Azéma, V. Leclère, W. van Ackooij).

Le problème : l'engagement (*commitment*) des groupes est décidé la veille et reste **fixe**,
tandis que le dispatch s'adapte séquentiellement à la demande nette observée. Seul le problème
racine est non convexe ; conditionnellement au planning, chaque étape est un programme linéaire.
On peut donc appliquer un SDDP paramétrique.

La contribution est une **formulation étendue par intervalles d'exploitation** qui produit des
coupes de programmation dynamique plus fortes que la représentation 3-bin standard.

Ce dépôt est un fork de [SDDP.jl](https://github.com/odow/SDDP.jl) (module renommé `MSUC`)
augmenté du modèle UC et du générateur de coupes intervalles.

---

## Installation

| | |
|---|---|
| Julia | **1.12.6** (version de référence du `Manifest.toml`) |
| Solveur | **Gurobi** avec licence valide (`GRB_LICENSE_FILE`) |

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using MSUC; println("ok")'
```

> **Julia 1.11** — le `Manifest.toml` versionné est résolu pour 1.12. Pour travailler en 1.11,
> supprimer `Manifest.toml` et relancer `Pkg.instantiate()`.

---

## Démarrage rapide

```julia
using MSUC, Gurobi
const GRB_ENV = Gurobi.Env()
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

T = 24
instance = MSUC.parse_nc4("Data/T-Ramp/10_0_1_w.nc4", optimizer, T)

model = MSUC.PolicyGraph(
    MSUC.subproblem_builder_UC_extended,   # ou subproblem_builder_UC_def pour la 3-bin
    instance,
    1.0,                                   # `force` : amplitude de l'aléa (0 => déterministe)
    MSUC.Graph(T + 1),                     # T+1 nœuds : racine + T étapes de dispatch
    MSUC.ExtendedFormulation,              # ou MSUC.SDDPFormulation
    lower_bound = 0.0,
    optimizer = optimizer,
)

MSUC.train(model; iteration_limit = 20)              # au moins un critere : iteration_limit et/ou time_limit
MSUC.train(model; iteration_limit = 20, mip_gap = 1e-2)   # ecart d'optimalite du MIP racine (defaut du solveur : 1e-4)
MSUC.train(model; iteration_limit = 20, passes_per_mip = 5)   # 5 passes avant/arriere (LP) par resolution du MIP racine (defaut : 1)
```

`mip_gap` agit sur le MIP racine, poste dominant du temps d'entraînement : un écart plus large l'accélère
sans invalider la borne inférieure, qui reste la borne duale du MIP (voir `train`).

`passes_per_mip` fait plusieurs passes avant/arrière à planning fixé (donc des LP seulement) avant de
re-résoudre le MIP racine : avec 1, c'est l'algorithme usuel. Les coupes restent valides, et seule celle de la
dernière passe est ajoutée à la racine. Une « itération » est alors un cycle complet (un MIP puis les passes).

`train` affiche son journal (une ligne par itération) uniquement dans le terminal, sans écrire de fichier ;
`print_level = 0` le coupe. Pour l'exploiter dans le code, `model.most_recent_training_results.log` contient
une entrée par itération (`bound`, `incumbent`, `time`, `total_solves`, ...).

La colonne `bound` du journal est la borne inférieure, issue du problème racine (26) résolu
avec l'approximation courante (Algorithme 2, ligne 5). La colonne `incumbent` est la valeur de l'objectif
du planning $x^k$ retenu par ce MIP : avec un `mip_gap` non nul elle est supérieure à `bound` (d'au plus
`mip_gap` en relatif) et n'est pas une borne.

> Le problème racine est un **MIP**. La borne publiée est sa borne **duale**
> (`objective_bound`), pas la valeur de l'incumbent : avec un `MIPGap` non nul, l'incumbent
> *majore* l'optimum du MIP et n'est donc pas une borne inférieure valide. La trajectoire de
> la passe avant utilise en revanche la solution incumbent — n'importe quel planning
> admissible convient pour engendrer des coupes. Voir `solve_first_stage_problem`.
>
> La borne duale est valide à chaque itération mais **pas monotone** (le branch-and-bound
> peut s'arrêter à un point moins avancé qu'à l'itération précédente). Comme chaque borne
> minore l'optimum vrai, c'est leur **maximum courant** qui est publié : valide et monotone.

### Comparer les trois solveurs

[`examples/compare_formulations.jl`](examples/compare_formulations.jl) lance les trois sur le
même problème **déterministe** (`force = 0`), où l'optimum exact est calculable et sert de
référence aux deux bornes SDDP :

```bash
julia --project=. examples/compare_formulations.jl
```

```julia
using MSUC, Gurobi, Printf
const GRB_ENV = Gurobi.Env()
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

T, FORCE, ITER = 24, 0.0, 10
instance = MSUC.parse_nc4("Data/T-Ramp/10_0_1_w.nc4", optimizer, T; S = 1)   # S = 1 : aléa déterministe

# 1. Formulation extensive : optimum exact du problème déterministe
res = MSUC.multistage_extensive(instance; force = FORCE, gap = 1e-6, timelimit = 300)
opt = res[1]

# 2. et 3. Les deux SDDP sur le même problème
function run_sddp(builder, method, label)
    model = MSUC.PolicyGraph(builder, instance, FORCE, MSUC.Graph(T + 1), method;
                             lower_bound = 0.0, optimizer = optimizer)
    MSUC.train(model; iteration_limit = ITER, print_level = 0)
    log = model.most_recent_training_results.log
    @printf("SDDP %-11s borne = %.1f  (%.1f %% de l'optimum)\n",
            label, log[end].bound, 100 * log[end].bound / opt)
    return [l.bound for l in log]
end

b_ext  = run_sddp(MSUC.subproblem_builder_UC_extended, MSUC.ExtendedFormulation, "intervalle")
b_3bin = run_sddp(MSUC.subproblem_builder_UC_def,      MSUC.SDDPFormulation,     "3-bin")

# Une borne inférieure valide ne peut pas dépasser l'optimum exact.
@assert b_ext[end] <= opt && b_3bin[end] <= opt
```

Sortie obtenue sur `10_0_1_w.nc4` (10 unités, 2025 intervalles, optimum exact **1 850 053,2**) :

| itération | intervalle (39) | % opt. | 3-bin (13) | % opt. |
|---|---|---|---|---|
| 1 | 58 914 | 3,2 | 58 914 | 3,2 |
| 4 | 389 064 | 21,0 | 349 193 | 18,9 |
| 7 | 1 116 737 | 60,4 | 351 265 | 19,0 |
| 9 | 1 808 431 | 97,8 | 352 061 | 19,0 |
| 10 | **1 809 147** | **97,8** | 352 376 | 19,0 |

C'est la thèse de l'article : les coupes intervalles atteignent 98 % de l'optimum en 9 itérations,
là où la 3-bin n'a pas dépassé 19 % après 10. Les trajectoires dépendent des solutions du MIP racine
(duaux dégénérés) : lisez ces chiffres comme un ordre de grandeur, pas comme une valeur à reproduire
à la décimale.


### Exemple jouet : encadrement complet de l'optimum

À `T = 24` la 3-bin ne converge pas et l'arbre de scénarios est hors de portée. À `T = 4`
avec `S = 2`, l'arbre n'a que $2^4 = 16$ feuilles : on calcule l'optimum multi-étapes
**exact** par son équivalent déterministe et on encadre sans rien supposer.
[`examples/toy_convergence.jl`](examples/toy_convergence.jl) :

```bash
julia --project=. examples/toy_convergence.jl
```

$$\underbrace{\text{borne inférieure}}_{\text{entraînement}} \;\le\; \underbrace{\text{optimum}}_{\text{arbre exact}} \;\le\; \underbrace{\text{coût de la politique}}_{\text{simulation}}$$

Optimum multi-étapes exact = **235 205,1131** (arbre de 31 nœuds) :

| itération | intervalle (39) | % opt. | 3-bin (13) | % opt. |
|---|---|---|---|---|
| 1 | 43 262,46 | 18,4 | 43 262,46 | 18,4 |
| 3 | 218 866,32 | 93,1 | 50 665,85 | 21,5 |
| **8** | **235 205,11** | **100,0** | 230 534,80 | 98,0 |
| **20** | 235 205,11 | 100,0 | **235 205,11** | **100,0** |

| | borne | politique | encadrement |
|---|---|---|---|
| intervalle (39) | 235 205,1131 | 235 205,1131 | $-3{,}7\cdot10^{-16}$ |
| 3-bin (13) | 235 205,1131 | 235 205,1131 | $5\cdot10^{-16}$ |

Les trois quantités coïncident à la précision machine, pour les deux formulations — la
formulation intervalle en **8 itérations contre 20**. C'est la vérification la plus
complète du dépôt : elle couvre la validité des coupes, l'équivalence de la
**Proposition 3**, le processus ARMA, l'état $\zeta$ et les coefficients $\tilde\delta$.

Trois briques la rendent possible :

- **`multistage_extensive(instance; force)`** — équivalent déterministe de l'arbre de
  scénarios, donc l'optimum **exact** du problème multi-étapes (1) avec la
  non-anticipativité (1j). L'arbre a $\sum_{t=0}^{T} S^t$ nœuds, donc n'est tractable que pour
  $S$ et $T$ petits ; avec une instance construite en `S = 1`, l'aléa est déterministe et on obtient
  l'optimum exact du problème déterministe. Les lignes (44) n'y sont pas imposées — sur l'horizon complet elles
  sont impliquées par les rampes — ce qui rend la référence indépendante de (44) et (46).
- **`simulate(model)`** — coût espéré de la politique entraînée, donc un
  **majorant** de l'optimum. L'engagement est figé (état sortant de la racine), les coupes
  intervalles réinstanciées à ce planning, puis chaque chemin est parcouru en résolvant les
  sous-problèmes d'étage. Avec `schedule = ...`, le planning donné est fixé directement, sans résoudre le
  MIP racine. Le coût d'étage exclut la variable d'épigraphe : on mesure le coût
  réellement encouru, pas l'approximation de la valeur future. Énumère les $S^T$ chemins
  quand c'est tractable (coût espéré **exact**, aucune erreur d'échantillonnage), sinon tire
  `samples` chemins avec une graine.
- **`scenario_tree(instance, T)`** — l'arbre lui-même. La récursion (1g) y est évaluée
  numériquement, donc $\zeta$ n'est pas une variable du programme étendu.

---

## Campagnes d'expériences SMS++ (`runSMS.jl`)

[`runSMS.jl`](runSMS.jl) entraîne puis simule la politique sur les instances de `Data/T-Ramp`, un run par
cœur (`Distributed`). Le script ne contient que la mise en place des workers et les deux points d'entrée
`run_train` et `run_simulate` ; le travail d'un run (`train_job`, `simulate_job`, écriture et relecture de la
politique) est dans [`src/functions/run/`](src/functions/run/). Depuis la racine du dépôt :

```julia
include("runSMS.jl")                      # NBWORKERS=2 julia ... pour un essai (10 workers par défaut)

size_list = [10, 50]                      # nombre de groupes de l'instance
id_list   = 1:5                           # identifiant (Data/T-Ramp/<n>_0_<id>_w.nc4)
methods   = ["interval", "bin3"]          # coupes intervalle ou 3-bin

# 1. Entraînement : un run de 3 600 s par combinaison (taille, identifiant, méthode, S, sigma, ...)
run_train(size_list, id_list, methods, [3600];
          S_list = [3, 5], sigma_list = [0.25, 1.0], mip_gap_list = [1e-2])

# 2. Politique telle qu'elle était à 60, 600 et 3 600 s, simulée sur 1 000 trajectoires
run_simulate(size_list, id_list, methods, [3600], [60, 600, 3600], 1000;
             S_list = [3, 5], sigma_list = [0.25, 1.0], mip_gap_list = [1e-2])
```

Les tailles disponibles sont 10, 20, 50, 75, 100, 150 et 200 ; les identifiants vont de 1 à 5, sauf pour
200 groupes (1 à 11). Une combinaison (taille, identifiant) sans fichier est ignorée avec un avertissement,
ce qui permet par exemple `size_list = [10, 200]` avec `id_list = 1:11`.

Les options `S_list`, `sigma_list`, `mip_gap_list` et `passes_per_mip_list` (valeurs par défaut `[5]`, `[1.0]`,
`[nothing]` et `[1]`) donnent un run par élément de leur produit cartésien avec les instances, les méthodes
et les temps ; `T` (24) est un scalaire. L'amplitude de l'aléa est réglée par `sigma`, qui multiplie
l'écart-type de l'innovation de l'ARMA (variance $2{,}9\cdot10^{-4}\sigma^2$) : il joue exactement le rôle du
paramètre `force` de `PolicyGraph`, fixé à 1 dans les runs, et `sigma = 0` redonne le problème déterministe. La graine est fixée à 1 pour tous les runs
(`MSUC.SEED`), entraînement et simulation. Pour `run_simulate`, les listes doivent contenir les valeurs de
l'entraînement (elles peuvent n'en être qu'un sous-ensemble).

`run_simulate` s'appelle après l'entraînement, éventuellement dans une autre session, avec d'autres
instants ou un autre nombre de trajectoires. À un instant donné, la politique rejouée est celle de la
dernière itération `k` terminée : son planning $x^k$ (celui du MIP racine) et les coupes d'étape qui
existaient quand il a été calculé, c'est-à-dire celles jusqu'à l'itération `k-1`. C'est la politique que
simule la passe avant de l'itération `k`, et la borne `bound` de cette itération repose sur ces mêmes
coupes. Un instant antérieur à la fin de la première itération est ignoré. `run_simulate` reconstruit
un modèle neuf, y recharge ces coupes d'étape (les instants sont traités par ordre croissant, sans rien
recharger), fixe le planning enregistré (`simulate(model; schedule = ...)`) puis simule `N` trajectoires :
le MIP racine n'est jamais re-résolu. Si `S^T ≤ 4096` les trajectoires sont toutes énumérées et le coût est exact. Les options
(`T`, `S`, `sigma`, `mip_gap`, `passes_per_mip`) doivent être celles de l'entraînement.

**L'entraînement n'est pas modifié** : `train` est appelé tel quel. On relève seulement, après chaque
itération, le planning du MIP racine et les temps par phase ; les coupes restreintes décrites ci-dessous
sont écrites une fois l'entraînement terminé.

**Pourquoi les coupes enregistrées sont petites, sans perte.** Évaluer une coupe en un planning ne demande
que ses coefficients sur les composantes que ce planning sélectionne : les intervalles choisis en
formulation intervalle, les composantes non nulles de $(u, v, w)$ en 3-bin. On ne garde donc, pour chaque
coupe, que ses coefficients sur les composantes sélectionnées par au moins un planning de
l'entraînement, soit 75 intervalles sur 2 025 après 53 itérations sur `10_0_1_w`. Chaque passe ajoute une
coupe à chaque nœud d'étape, ce qui donne les coupes qui existaient à chaque itération sans rien relever
de plus. Les coupes de la racine ne sont pas enregistrées, puisque le MIP racine n'est pas re-résolu. Sur `10_0_1_w` (`T = 24`) la politique
rejouée est identique, au bit près, à celle obtenue avec les coupes complètes.

Chaque run écrit, dans `results_sms/<méthode>_T<T>_S<S>_sigma<σ>_passes<p>_gap<g>/<instance>/time_<t>/` :

| Fichier | Contenu |
|---|---|
| `log.csv` | une ligne par itération : `bound` (borne inférieure), `incumbent` (valeur de l'incumbent du MIP racine : égale à `bound` si le MIP est résolu à l'optimalité, plus grande sinon), `time` (s depuis le début), `total_solves` |
| `states.csv` | états échantillonnés par les passes avant : une ligne par (itération, passe, nœud), colonnes `power_<unité>` et `zeta_<i>` = état $(y,\zeta)$ à la sortie du nœud ; le nœud 1 est l'état initial ; `passes_per_mip` passes par itération |
| `phases.csv` | une ligne par itération : temps **cumulé** (s) de chaque phase, une colonne par phase ; le temps d'une phase entre deux instants est une différence de deux lignes |
| `timers.csv` | bilan final : phase, nombre d'appels, temps |
| `config.csv` | paramètres du run, nombre d'itérations, temps total, dernière borne |
| `schedules.csv` | une ligne par itération : intervalles sélectionnés par le MIP racine (`selected`) et composantes non nulles de $x=(u,v,w)$ (`x_nonzero`), listes séparées par des points-virgules |
| `used_components.csv` | composantes de $x$ utiles à au moins un planning (voir ci-dessous) |
| `cuts.csv` | coupes d'étape restreintes à ces composantes, sans en-tête, une ligne par coupe dans l'ordre de création : nœud, $\alpha$, $\delta$, $\tau$, $\beta$ |
| `simulation_<N>.csv` | un résumé par instant : dernière itération terminée, sa borne, nombre de coupes d'étape, coût espéré de la politique, coût d'engagement, erreur standard et intervalle de confiance à 95 % (`ci95_low`, `ci95_high` : coût espéré $\pm\,1{,}96$ erreur standard ; nul si les trajectoires sont énumérées, le coût est alors exact) |
| `simulation_<t>_<N>.csv` | coût de dispatch de chaque trajectoire à l'instant `t` |

Les phases de `phases.csv` sont `solve_first_stage_problem` (le MIP racine), `set_schedule`,
`forward_pass` et `backward_pass`, avec leurs sous-phases (`backward_pass/compute_coefficients`,
`backward_pass/solve_subproblem`, `backward_pass/add_cut`, ...).

`GurobiThreads` (1 par défaut) est le nombre de fils de Gurobi par run : gardez
`Nbworkers × GurobiThreads` inférieur au nombre de cœurs physiques, le temps étant la grandeur mesurée.
Sur `10_0_1_w` (`T = 24`), `cuts.csv` pèse environ 1,9 ko par coupe en formulation intervalle (contre 21 ko
pour les coupes complètes, en binaire) et 3,3 ko en 3-bin, soit 2 à 3 Mo après une vingtaine de secondes. Il
croît avec le nombre de composantes utiles, donc avec la taille de l'instance et la durée du run.
`states.csv` pèse environ 100 octets par ligne pour 10 unités (une ligne par itération, passe et nœud, donc
`passes_per_mip × (T+1)` lignes par itération) et croît proportionnellement au nombre d'unités.
`results_sms/` est ignoré par git.

---

## Incertitude : processus ARMA

L'erreur relative de prévision de demande nette suit le processus ARMA de la §2.1, avec les
paramètres calibrés de l'article PSCC, éq. (40) :

$$\eta_t = 0{,}67\,\eta_{t-1} + \xi_t + 0{,}117\,\xi_{t-1}, \qquad \xi_t \sim \mathcal N(0,\ 2{,}9\cdot10^{-4}\sigma^2)$$

Ici $\sigma$ multiplie l'**écart-type** de l'innovation (le paramètre $\sigma$ de PSCC multiplie la variance : c'est
le carré de celui-ci) ; $\sigma = 1$ est le cas de base et $\sigma = 0$ le problème déterministe.

Un **seul** processus pilote tous les nœuds du réseau : l'erreur relative $\eta_t$ est la même
partout (la demande du nœud $n$ vaut $d_{n,t}\,(1+\eta_t)$, comme une charge nette de réseau répartie au
prorata des charges). Le processus est rendu markovien par l'augmentation d'état de la §2.1 :
$\zeta_t = (\eta_t, \xi_t)$ pour un ARMA(1,1), donc de **dimension $p+q = 2$ quel que soit le réseau**
(118 nœuds compris), avec

$$\zeta_t = \Psi\,\zeta_{t-1} + E\,\xi_t, \qquad \eta_t = S_\eta\,\zeta_t$$

où $S_\eta$ ne fait que recopier $\eta_t$ sur les nœuds. $\zeta$ est une **variable d'état** du graphe,
déclarée à tous les nœuds, et les coupes portent le terme $\tilde\delta_t^\top\zeta$ des éq. (10), (12) et (40b).

> ⚠️ **Collision de notation entre les deux articles.** Ce que PSCC (40) note $\eta_t$ est
> l'**innovation**, notre $\xi_t$ ; ce que PSCC note $\varepsilon_t$ est l'**erreur**, notre
> $\eta_t$. Le code suit la notation MSUC.

### Régler le support

PSCC spécifie une innovation **gaussienne continue**, mais l'Hypothèse 1 exige un support
**fini**. La discrétisation n'est donnée dans aucun des deux articles ; elle est ici un
paramètre du parsing :

```julia
instance = MSUC.parse_nc4("Data/T-Ramp/10_0_1_w.nc4", optimizer, 24;
                          sigma          = 1.0,              # multiplie l'écart-type de l'innovation
                          S              = 5,                # taille du support
                          discretization = :gauss_hermite)   # ou :equiprobable
```

Les défauts sont les constantes `ARMA_SUPPORT_SIZE` et `ARMA_DISCRETIZATION` de
[`src/functions/instance/arma_process.jl`](src/functions/instance/arma_process.jl). Le nombre d'aléas par étape vaut
exactement `S` : un seul facteur ARMA scalaire pilote la demande de tous les nœuds (même erreur relative
partout, `S_eta` la recopie sur les nœuds), sans quoi il faudrait $S^{|\mathcal N|}$ combinaisons. L'état
d'incertitude $\zeta$ est donc de dimension $p+q$ (2 pour l'ARMA(1,1)), quel que soit le réseau.

**Les deux schémas**, tous deux de moyenne et variance exactes :

| | $S=5$, nœud extrême | $p_{\min}$ | moments exacts |
|---|---|---|---|
| `:gauss_hermite` (défaut) | $\pm 2{,}86\,\sigma_\xi$ | 1,1 % | jusqu'à l'ordre $2S-1$ |
| `:equiprobable` | $\pm 1{,}48\,\sigma_\xi$ | 20 % | moyenne et variance |

- **Gauss-Hermite** (Golub-Welsch : valeurs propres de la matrice de Jacobi d'Hermite,
  poids = carrés des premières composantes des vecteurs propres). Optimal pour une
  **espérance** — c'est le cas risque-neutre du papier. Revers : les nœuds extrêmes portent
  des probabilités infimes ($8{,}6\cdot10^{-10}$ à $S=15$), donc les queues comptent peu.
- **Équiprobable** : chaque scénario pèse $1/S$, nœud = moyenne conditionnelle de sa strate.
  À préférer pour une **mesure de risque** type AVaR, où les queues doivent être peuplées.

> Le schéma équiprobable brut **sous-estime la variance** (0,897 à $S=5$) : la dispersion
> intra-strate est perdue. Les nœuds sont donc renormalisés par défaut
> (`match_variance = true`) pour que la variance discrète vaille exactement
> $2{,}9\cdot10^{-4}\sigma^2$. Sans effet sur Gauss-Hermite, déjà exacte.

`force` (argument de `PolicyGraph`) multiplie $\eta_t$ dans le bilan (1h) : `force = 0`
neutralise l'aléa et, comme $\tilde\delta \equiv 0$ par récurrence, redonne **exactement**
le problème déterministe — c'est ce qui rend le protocole de validation reproductible. `sigma` a le même
effet que `force` : multiplier l'écart-type de l'innovation par $c$ revient à prendre `force` $= c$.

---

## Instance IEEE 118-bus

```julia
instance = MSUC.parse_IEEE("118_syst", optimizer, 24; S = 5)   # sigma, discretization : comme parse_nc4
model = MSUC.PolicyGraph(MSUC.subproblem_builder_UC_extended, instance, 1.0, MSUC.Graph(25),
                         MSUC.ExtendedFormulation; lower_bound = 0.0, optimizer = optimizer)
MSUC.train(model; iteration_limit = 50, mip_gap = 1e-2)
```

`parse_IEEE` (dans [`parse_instance.jl`](src/functions/instance/parse_instance.jl)) lit
`Data/IEEE/<dossier>` : 54 groupes thermiques, 118 nœuds, réseau en flux DC (chaque ligne dans les
deux sens, capacité et réactance du fichier), horizon `T ≤ 24`.

- **Demande** : charge maximale de chaque nœud (`maximum_load.csv`) × profil horaire
  (`load_distribution_profile.csv`, en %). L'éolien n'est **pas** modélisé : la demande nette est la
  charge.
- **Incertitude** : le facteur ARMA(1,1) commun décrit plus haut, donc un état $\zeta$ de dimension 2
  et exactement `S` aléas par étape. Les paramètres sont ceux de PSCC (40), appliqués tels quels à ce réseau ;
  $\sigma$ est le même à toutes les heures.

Validé comme le toy, à `T = 4`, `S = 2` (12 itérations, `mip_gap = 1e-3`), face à l'optimum multi-étapes
**exact** de l'arbre (`multistage_extensive`, 146 328,4058) :

| | borne | % opt. | politique | borne ≤ opt. ≤ politique |
|---|---|---|---|---|
| intervalle (39) | 143 449 | 98,0 | 149 273 | oui |
| 3-bin (13) | 3 136 | 2,1 | 363 518 | oui |

Les deux encadrements sont valides ; en revanche la 3-bin reste très en dessous de l'optimum sur ce
réseau, là où les coupes intervalles en sont à 98 %. À `T = 24`, `S = 5` l'entraînement tourne
(quelques secondes par itération avec `mip_gap = 1e-2`), mais aucun optimum exact n'existe pour
comparer : la validation de cette taille repose sur le test à `T = 4`.

---

## Correspondance article ↔ code

C'est la table de référence pour lire le code.

### Données et ensembles

| Article | Code | Où |
|---|---|---|
| $\mathcal J$ — groupes | `instance.Thermalunits` | `struct_functions/instance/instance.jl` |
| $\mathcal N$ — nœuds du réseau | `instance.Next`, `instance.Demandbus` | idem |
| $\mathcal L$ — lignes | `instance.Lines` | idem |
| $T$ — horizon | `instance.TimeHorizon` | idem |
| $\mathcal T_j$ — intervalles admissibles | `unit.intervals`, vecteurs `[a, b]` | `set_intervals` |
| $\underline P_j,\ \overline P_j$ | `unit.MinPower`, `unit.MaxPower` | |
| $R^U_j,\ R^D_j$ | `unit.DeltaRampUp`, `unit.DeltaRampDown` | |
| $S^U_j,\ S^D_j$ | `unit.MinPower` (ce modèle pose $S^U = S^D = \underline P$) | |
| $y_{j,0}$ | `unit.InitialPower` | |
| $\xi_t$ — aléa (support fini) | `instance.arma.support` | `struct_functions/instance/arma_process.jl` |
| $p_{t,\xi}$ | `instance.arma.probability` | idem |
| $\Psi,\ E,\ S_\eta$ — (1g) ; $\zeta_0$ | `instance.arma.Psi`, `.E`, `.S_eta` ; `.zeta0` | idem |

### Variables

| Article | Code |
|---|---|
| $x_j = (u_j, v_j, w_j)$ | variables binaires `u`, `v`, `w` de la racine (`RootVariables`) ; **paramètres** `StageParameters` aux étapes de dispatch, fixés par `set_schedule!` ; valeurs lues dans `Schedule` |
| $y_{j,t}$ | `power[j]` (état), `node.power` |
| $\zeta_t$ | `ζ` (état), `node.zeta` |
| $\gamma_{j,a,b}$ | `RootVariables.gamma[j,a,b]` à la racine ; `Schedule.selected` (intervalles choisis) ensuite |
| $\tilde y_{j,a,b,t}$ | jamais matérialisé : identité $\tilde y = u_{a,b,t}\, y_{j,t}$, cf. (41g) |
| $f_{n,m,t},\ \theta_{n,t}$ | `flow`, `θ` |

### Section 4 — modèle capacité + rampes (`functions/instance/thermal_unit.jl`)

| Article | Code |
|---|---|
| $u_{a,b,t}$ — éq. (15a) | `is_on_interval(a, b, t)` |
| $\underline p_{j,a,b,t}$ | `p_lower(unit, a, b, t)` |
| $\bar p_{j,a,b,t}$ — inclut (46) | `p_upper(unit, a, b, t, T)` |
| $R^U_{j,a,b,t}$ — membre droit de (45b) | `ramp_up(unit, a, b, t)` |
| $R^D_{j,a,b,t}$ — membre droit de (45c) | `ramp_down(unit, a, b, t)` |
| $\tilde y_{j,a,b,0}$ — éq. (17) | `initial_interval_output(unit, a)` |
| $q_{j,a,b,t}$ — éq. (49) | `reference_output(unit, a, b, t, T)` |
| $\tilde y^k_{j,a,b,t}$ — règle (55) | `linearization_point(unit, a, b, t, T, y)` |
| $\hat Q_{j,a,b,t}$ et son support (48)–(49) | `local_support(unit, a, b, t, T, y_prev, z)` → `(h, β)` |

### Coupes (`struct_functions/cuts/cuts.jl`, `functions/cuts/cuts.jl`, `functions/cuts/refine_cuts.jl`)

| Article | Code |
|---|---|
| coupe 3-bin (12) | `BinCut` : `alpha`, `tau` (sur le vecteur plat $x$), `beta` (sur $y$), `delta` (sur $\zeta$) |
| coupe intervalle (41) | `IntervalCut` |
| $\tilde\alpha_t$ — (39a) | `IntervalCut.alpha` |
| $\tilde\tau_{j,a,b,t}$ | `IntervalCut.tau[id]`, `id` = identifiant plat de l'intervalle (`instance.intervals`, `IntervalIndex`) |
| $\tilde\beta_{j,a,b,t}$ | `IntervalCut.beta[id]` |
| $V^{\text{ext}}_{t}$ | `node.cuts` |
| coupe au planning courant | `instantiate(cut, schedule, N)` → $(A, b)$ tels que $\theta \ge A + \delta^\top\zeta + b^\top y$ |
| $\hat z_{j,a,b,\xi}$ — Prop. 4 | `z` |
| $(\hat h, \hat\beta)$ — (34), (49) | valeur de retour de `local_support` |

### Multiplicateurs

**Attention à ce point** : le code nommait `μ` ce qui est le $\sigma$ de l'article.

| Article | Code | Nature |
|---|---|---|
| $\sigma \in \Delta_k$ — (41g) | `σ` | duaux des lignes d'épigraphe, un par coupe |
| $\nu_{n,t}$ — (41f) | `ν` | duaux du bilan par nœud |
| $\mu_{j,a,b}$ — (42b) | *jamais matérialisé* | la forme close (49) donne directement le support |

---

## Comment l'Algorithme 2 se lit dans le code

| Alg. 2 | Code |
|---|---|
| l. 4–5 — résoudre (19), stocker $(\gamma^k, x^k)$ et $L^k_{\text{ext}}$ | `solve_first_stage_problem` (MIP racine, borne duale) ; `extract_schedule` ; `set_schedule!` — appelés dans `one_iteration` |
| l. 6–11 — passe avant | `forward_pass` |
| l. 9 — résoudre le sous-problème d'étape | `solve_subproblem` (planning fixé par `set_schedule!`) |
| l. 10 — état d'intervalle | identité $\tilde y = u_{a,b,t} y_{j,t}$, pas de variable dédiée |
| l. 12–20 — passe arrière | `backward_pass` → `solve_next_node` → `refine_bellman_function` |
| l. 15–16 — duaux $(\hat\nu, \hat\sigma)$ | `get_dual_solution` → `StageDuals` |
| l. 17 — $\hat z_{j,a,b,\xi}$ | `compute_coefficients_extended` |
| l. 19–28 — support local | `local_support` |
| l. 29, (36) — $\hat\tau = \hat h + \tilde\tau^\sigma$ | `compute_coefficients_extended` : `_weighted_sum` (propagation $\tilde\tau^\sigma$, tous les intervalles) puis boucle sur les intervalles actifs (valeur locale $\hat h$) → `τ[id]` |
| l. 33–34, (39)–(40) — moyenne et ajout | `IntervalCut(α, τ, β, δ, nothing)`, puis `add_cut!` |

**Le point le plus subtil.** Les coupes sont stockées avec leurs coefficients *par intervalle*
(*par composante de x* en 3-bin), puis la ligne d'épigraphe est **réécrite** au planning
courant à chaque résolution du MIP racine (`set_schedule!`, appelée dans `one_iteration` →
`instantiate_cut!`, par `set_normalized_rhs` / `set_normalized_coefficient`). C'est ainsi qu'on obtient (41g) sans porter les produits
$\gamma \cdot \tilde y$ comme variables.

**Le piège à ne pas réintroduire.** La Prop. 4 annule $\hat h$ et $\hat\beta$ sur un intervalle
inactif, mais l'éq. (36) ajoute ensuite $\tilde\tau^\sigma$ : la **propagation des coupes futures
a lieu pour tous les intervalles**, y compris inactifs. Un intervalle qui démarre plus tard est
inactif à l'étape courante mais porte bien une valeur future. L'omettre rend la borne invalide
(mesuré : 1 905 374 pour un optimum exact de 1 850 053).

---

## Protocole de validation

Avec `force = 0.0` le problème devient **déterministe**, donc l'optimum exact est calculable par
la formulation extensive. La borne SDDP doit alors rester **sous** cet optimum, croître, et
converger dessus. C'est le seul test qui tranche.

```julia
# instance = MSUC.parse_nc4(...; S = 1)  (aléa déterministe)
r = MSUC.multistage_extensive(instance; force=0.0, gap=1e-6, timelimit=120)
opt = r[1]          # 1 850 053.2 sur Data/T-Ramp/10_0_1_w.nc4

model = MSUC.PolicyGraph(MSUC.subproblem_builder_UC_extended, instance, 0.0,
                         MSUC.Graph(25), MSUC.ExtendedFormulation;
                         lower_bound=0.0, optimizer=optimizer)
MSUC.train(model; iteration_limit = 10)
```

Référence (instance `10_0_1_w.nc4`, optimum exact **1 850 053,2**) :

| itération | 1 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| borne | 58 914 | 389 064 | 433 010 | 1 428 974 | **1 809 147** (97,8 %) |

> ⚠️ **Ne comparez pas** la borne étendue à celle de la 3-bin pour juger de la validité : tant
> que la 3-bin n'a pas convergé, l'étendue est légitimement bien plus haute — c'est exactement
> ce que l'article prédit. Seule la comparaison à un optimum exact est concluante.

> ⚠️ En stochastique (`force > 0`), la passe avant tire ses scénarios dans le générateur aléatoire
> global : sans `Random.seed!(...)` avant `train`, deux exécutions donnent des bornes différentes
> (l'exemple toy et `runSMS.jl` fixent la graine). `simulate` a sa propre graine (`seed`). Pour une comparaison A/B de
> code, `force = 0` reste le plus simple.

---

## Organisation du dépôt

Le code est rangé en deux dossiers parallèles, avec les mêmes sous-dossiers : `functions/` contient
les fonctions, `struct_functions/` les structures dont les fichiers de `functions/` ont besoin
(même nom de fichier). `MSUC.jl` inclut d'abord toutes les structures, puis les fonctions.

```
src/
  MSUC.jl                          module, imports, constantes de coût
  functions/
    algorithm/                     la boucle SDDP (Algorithme 2)
      train.jl                       train, master_loop
      one_iteration.jl               une itération : racine, passe avant, passe arrière, borne
      solve_first_stage_problem.jl   problème racine (MIP) : planning x^k et borne inférieure
      forward_pass.jl                passe avant (étapes de dispatch)
      backward_pass.jl               passe arrière, solve_next_node
      solve_subproblem.jl            résolution d'un sous-problème d'étape, reprise numérique
    cuts/                          coupes de Benders
      refine_cuts.jl                 génération des coupes 3-bin et intervalle
      cuts.jl                        instantiate, add_cut!, ligne d'épigraphe de la racine
      duals.jl                       duaux d'un sous-problème (StageDuals)
    model/                         construction du modèle SDDP
      build_policy_graph.jl          PolicyGraph, @stageobjective, parameterize
      build_subproblems.jl           points d'entrée des deux formulations, états (y, ζ)
      build_root_problem.jl          problème racine : engagement et variables γ
      build_stage_problem.jl         problème d'étage : capacité/rampes, réseau, bilan, ARMA
      bellman_function.jl            variable θ et coupe initiale de chaque nœud
      schedule.jl                    extract_schedule, set_schedule!, paramètres du planning
      state_variable.jl              extension JuMP `State`
    instance/                      données et physique des unités
      parse_instance.jl              parse_nc4 (SMS++), parse_IEEE (118 nœuds)
      arma_process.jl                facteur ARMA commun et discrétisation de l'innovation
      thermal_unit.jl                Section 4 : T_j, p, R^U, R^D, q, local_support
      unit_constraints.jl            contraintes partagées (engagement, capacité/rampes, lignes)
    run/                           campagnes d'expériences (voir `runSMS.jl`)
      model.jl                       METHODS, dossier de résultats d'un run, optimiseur enregistré par worker, modèle neuf
      record.jl                      relevé des temps par phase et des plannings pendant l'entraînement
      policy.jl                      écriture et relecture de la politique (plannings, coupes restreintes)
      jobs.jl                        train_job, simulate_job, save_run
    reference/                     solutions exactes de référence
      multistage_extensive.jl        équivalent déterministe de l'arbre (optimum multi-étapes)
    tools/                         outils
      sampling.jl                    tirage des trajectoires et des aléas
      simulate.jl                    simulate : résout la racine puis les trajectoires
      print_log.jl                   journal et rapport numérique
  struct_functions/                mêmes sous-dossiers : Instance, ThermalUnit, ARMAProcess, Schedule,
                                   Node, PolicyGraph, BinCut, IntervalCut, Options, Log, ...
examples/
  toy_convergence.jl               toy T = 4 : encadrement borne ≤ optimum exact ≤ politique
  compare_formulations.jl          extensive / intervalle / 3-bin sur un problème déterministe
runSMS.jl                          campagne SMS++ : `run_train` et `run_simulate`, répartis sur des workers (le travail d'un run est dans `src/functions/run/`)
validation.ipynb                   notebook : entraînement, simulation, courbes de borne et de temps du MIP racine
toy_une_unite.ipynb                notebook : une unité (capacité 10, coût 1, coût fixe 1, rampes 1), délestage à coût 2, demande 15 ; bornes intervalle contre 3-bin
Data/
  T-Ramp/                          41 instances SMS++ (.nc4)
  IEEE/118_syst/                   données IEEE 118-bus (lues par `parse_IEEE`)
  Uncertainty/                     anciens scénarios de demande nette, plus lus par le code
```

Le **Benders deux-étages** de l'article [12] (`benders_callback`, `add_cut_SP`, l'heuristique
duale, les options `UCOptions`) a été retiré : c'est un autre algorithme, hors du périmètre de
ce papier. La référence exacte est `multistage_extensive`, aussi bien en déterministe (`S = 1`) qu'en stochastique.

Le code mort hérité de SDDP.jl a été supprimé : `Inner.jl`, `Experimental.jl`, `MSPFormat.jl`,
`biobjective.jl`, `binary_expansion.jl`, `cyclic.jl`, `alternative_forward.jl`,
`deterministic_equivalent.jl`, `modeling_aids.jl` et tout `visualization/`. Les dépendances
correspondantes (`HTTP`, `SHA`, `XLSX`, `RecipesBase`, `Plots`) sont sorties de `Project.toml`.

Cinq fonctionnalités de SDDP.jl ont également été retirées, **inatteignables** avec un
`LinearGraph` :

| | à quoi ça servait | pourquoi inatteignable |
|---|---|---|
| `belief_state` | graphes partiellement observables (POMDP) : loi de probabilité sur la partition de nœuds, mise à jour bayésienne, terme de coupe concave $\langle b,\mu\rangle$ | `belief_partition` n'est rempli que par `add_ambiguity_set` ; `LinearGraph` le laisse vide |
| `objective_state` | objectif dépendant d'un prix exogène markovien, avec interpolation et terme $\mu^\top y$ | positionné seulement par `add_objective_state`, que les builders n'appellent jamais |
| cycles | horizon infini : `terminated_due_to_cycle`, `starting_states`, `cycle_discretization_delta` | un chemin ne boucle pas |
| persistance de politique | `write_cuts_to_file` / `read_cuts_from_file` : sauver une politique entraînée pour la recharger sans réentraîner | ne sérialisent que les coupes génériques, **pas** les `IntervalCut` ; le dump était vide en formulation étendue, et `read_` jamais appelée |
| handlers de dualité (SDDiP, `ContinuousConicDuality`) et relaxation d'intégrité | `LagrangianDuality`, `StrengthenedConicDuality`, `BanditDuality`, `FixedDiscreteDuality`, `local_improvement_search.jl`, et la relaxation des variables entières avant la passe arrière | §1 du papier — seule la racine est non convexe, chaque étage est un LP : les duaux LP sont exacts, d'où `solve_subproblem(...; compute_duals = true)` |

`train` a aussi été réduit à l'essentiel : le multi-cut, les règles d'arrêt autres que `iteration_limit` et
`time_limit`, les schémas d'échantillonnage, le cache de la passe arrière et le fichier de journal ont été retirés.

> **`Plots` n'est plus une dépendance du projet.** Si vos notebooks font `using Plots`,
> rajoutez-le : `julia --project=. -e 'using Pkg; Pkg.add("Plots")'`.

---

## Ce qui n'est pas implémenté

| Article | État |
|---|---|
| §3.4 — dominance par étape et par itération | vide dans l'article |
| Horizon $T \neq 24$ | le builder et le parsing prennent `T`, mais les instances fournies sont à 24 pas |

Autres limites connues :

- **Pas d'option de graine dans `train`** : fixer `Random.seed!(...)` avant l'appel pour reproduire un run stochastique.
- **IEEE** : éolien non modélisé, $\sigma$ constant sur les 24 heures, pas d'optimum exact à `T = 24` pour valider.
- **Coût du calcul des coupes en $O(k)$** : les coefficients $\tilde\tau$ et $\tilde\beta$ sont stockés
  en vecteurs plats sur les intervalles et les sommes sur les $k$ coupes stockées se font par
  vecteurs entiers (`compute_coefficients_extended`), ce qui reste linéaire en $k$ par itération
  mais environ 14 fois plus rapide qu'avec des dictionnaires (T = 24, 10 unités, 15 itérations :
  3,4 s → 0,24 s). Le MIP racine domine désormais le temps d'entraînement. La parcimonie de la
  Remarque 7 n'est pas exploitée (tous les intervalles sont stockés).
- **Pas de tests automatisés.**
