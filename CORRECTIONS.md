# Corrections MSUC — suivi

Document de travail : audit du code vis-à-vis de l'article
*« Stronger cuts for SDDP for MSUC problems based on interval variables »*.
À mettre à jour au fur et à mesure des corrections.

**Légende des statuts** : `TODO` · `EN COURS` · `FAIT` · `VÉRIFIÉ` (corrigé **et** couvert par un test) · `NON-BUG`

Convention : quand un point change de statut, ajouter une ligne dans le [Journal](#journal).

---

## Tableau de bord

| ID | Sujet | Statut |
|----|-------|--------|
| B1 | `compute_delta_down_j_a_b` renvoyait le ramp up | **FAIT** |
| N1 | Point de linéarisation `incoming_state_j_a_b` | NON-BUG |
| N2 | Domaine local vide | NON-BUG |
| B2b | État $\tilde y_{j,a,b,0}$ dans la coupe racine, éq. (17) | **FAIT** |
| A1 | Alignement sur la §4 réécrite ($\bar p$, $q$, intervalles actifs) | **FAIT** |
| A2 | Propagation $\tilde\tau^\sigma$ des intervalles inactifs | **FAIT** |
| B4 | Bloc `gamma` dupliqué au nœud 1 | **FAIT** |
| B5 | `parse_IEEE` ne s'exécutait pas | **FAIT** (non validé de bout en bout) |
| B6 | `calculate_bound` désactivée (`bound = 0.0`) | **FAIT** |
| B7 | Incompatibilité Julia 1.12 (MbedTLS ← HTTP) | **FAIT** |
| E2 | Lignes de shutdown-feasibility (44) côté 3-bin | **FAIT** |
| E3 | Bruit non aligné entre les deux builders | **FAIT** |
| R1 | Refonte du calcul de $\hat Q_{j,a,b,t}$ | **FAIT** |
| NOT | Alignement de la notation sur l'article | **FAIT** |
| E1 | Processus ARMA $\zeta$ (1g) | **FAIT** — ARMA(1,1) de PSCC (40), état $\zeta$, coefficient $\tilde\delta$ |
| S1 | Aucun test automatisé | **PARTIEL** — `examples/toy_convergence.jl` encadre l'optimum exact (borne / arbre / politique) |
| B8 | Convention $b = T+1$ absente de `set_intervals` | **FAIT** |
| B9 | Borne = incumbent du MIP racine au lieu de sa borne duale | **FAIT** |
| S4 | Runs stochastiques non reproductibles (pas de graine) | TODO |
| P1 | Coût en $O(k)$ sur $\hat z$ → entraînement en $O(k^2)$ | PARTIEL |
| S3 | Nettoyage du dépôt | **FAIT** (hors artefacts versionnés) |
| S5 | Fonctionnalités SDDP.jl inatteignables (`belief_state`, `objective_state`, cycles, handlers SDDiP) | **FAIT** |

Détail des points historiques (B1, N1, N2, A1, A2) plus bas ; le README porte désormais la
table de correspondance article ↔ code.

---

## Protocole de validation : le cas déterministe

**C'est le test de référence.** Avec `force = 0.0` le problème devient déterministe, donc
l'optimum exact est calculable par la formulation extensive, et la borne SDDP doit
(i) rester **en dessous**, (ii) **croître**, (iii) **converger dessus**.

```julia
# vérité terrain
r = MSUC.bin_extensive_neutral(instance; silent=true, force=0.0, S=1, batch=1,
                               gap=1e-6, timelimit=120)
opt = r[3]            # 1_850_053.2  sur Data/T-Ramp/10_0_1_w.nc4

# borne étendue sur le MÊME problème
model = MSUC.PolicyGraph(MSUC.subproblem_builder_UC_extended, instance, 0.0,
                         MSUC.LinearGraph(25), MSUC.ExtendedFormulation;
                         sense=:Min, lower_bound=0.0, optimizer=optimizer)
MSUC.train(model; iteration_limit = 10)
```

Instance `Data/T-Ramp/10_0_1_w.nc4`, `T=24`, **optimum exact = 1 850 053,2**.

| itération | LB (B1 buggé) | LB (B1 corrigé) |
|---|---|---|
| 4 | 383 832 | 383 520 |
| 5 | 1 543 722 | 1 796 916 |
| 8 | 1 759 516 | 1 807 889 |
| 10 | 1 765 926 — **95,5 %** | **1 817 736 — 98,3 %** |

> **Le générateur de coupes étendu est valide.** La borne reste sous l'optimum, croît, et
> converge dessus. Correction de B1 = **+52 000** sur la borne à l'itération 10.

### ⚠️ Piège d'interprétation (erreur commise lors de l'audit initial)

Comparer la borne étendue à celle de la 3-bin **ne prouve rien** tant que la 3-bin n'a pas
convergé. Observé : extended 649 863 à 5 itérations vs 3-bin 380 160 à 60 itérations — c'est
exactement ce que l'article prédit (les coupes intervalles sont bien plus fortes), **pas** un
signe d'invalidité. Seule la comparaison à un **optimum exact** tranche.

⚠️ `train` **ajoute** ses logs à `MSUC.log`, qui est versionné (cf. [S3](#s3--nettoyage-du-dépôt)).

---

## Corrigé

### B1 — `compute_delta_down_j_a_b` renvoyait le ramp *up*

**Statut** : **FAIT** (non commité) · **Fichier** : `src/UC/Unit/Thermal_unit.jl:156`

```julia
else
    return unit.DeltaRampDown   # était unit.DeltaRampUp
```

C'est le $R^D_{j,a,b,t}$ de la **Proposition 6**. Dérivation depuis la ligne (43c) telle
qu'écrite dans le builder :
```
power.in - power.out <= -RD*start_down + (Pmin+RD)*is_on[t-1] - Pmin*is_on[t]
```
pour `a < t < b` : `start_down=0`, `is_on[t-1]=is_on[t]=1` ⇒ RHS $= P_{\min} + R^D - P_{\min} = R^D$.
Donc $\tilde y_t \ge \tilde y_{t-1} - R^D$ : c'est bien `DeltaRampDown`.

`RU ≠ RD` pour **les 10 unités** de l'instance (ex. unité 1 : 42,00 vs 32,60), donc l'erreur
mordait partout.

**Effet mesuré** : pas d'invalidité — le domaine local était simplement **mal dimensionné**, ce
qui **affaiblissait** les coupes. Borne à l'itération 10 : 1 765 926 → 1 817 736 (95,5 % → 98,3 %).

**Reste à faire** : test unitaire (cf. [S1](#s1--aucun-test)).
```julia
u = inst.Thermalunits[1]                      # RU=42.00, RD=32.60
@assert MSUC.compute_delta_down_j_a_b(u,5,15,7) == u.DeltaRampDown
@assert MSUC.compute_delta_up_j_a_b(u,5,15,7)   == u.DeltaRampUp
```

---

## Alignement sur la nouvelle Section 4 (2026-09-13)

### A1 — Alignement sur la nouvelle section 4

**Statut** : **FAIT** (non commité)

La §4 réécrite sépare enfin trois objets que le code confondait. `compute_Pmax_j_a_b` et
`incoming_state_j_a_b` étaient des copies l'une de l'autre ; ce sont maintenant deux choses
distinctes.

| Objet | Éq. | Fichier | Changement |
|---|---|---|---|
| $\bar p_{j,a,b,t}$ | (46) | `Thermal_unit.jl` | `compute_Pmax_j_a_b` : **cône de démarrage retiré** — il appartient à $q$, pas à $\bar p$ |
| $q_{j,a,b,t}$ | (49) | `Thermal_unit.jl` | `incoming_state_j_a_b` : garde le cône ; en $t=0$, état initial **(17)** |
| $R^U, R^D_{j,a,b,t}$ | (45b)(45c) | `Thermal_unit.jl` | inchangés — déjà corrects après [B1](#b1--compute_delta_down_j_a_b-renvoyait-le-ramp-up) |
| $\tilde y_{j,a,b,0}$ | (17) | `bellman_functions.jl` | coupe racine **et** `intercept_cstr` : `InitialPower * (a<=0)` |
| intervalle actif | Prop. 4 | `bellman_functions.jl` | $\hat h = \hat\beta = 0$ si $t+1 \notin [a:b-1]$ |
| tous les intervalles participent | (27) | `bellman_functions.jl`, `forward_passes.jl` | `unit.intervals` partout ; filtres `t+1<=b` supprimés |

`intervals_t_extended` n'est **plus utilisé** dans le générateur de coupes (le champ reste dans
`ThermalUnit` et `parse_nc4` le calcule encore — à nettoyer).

⚠️ La généralisation « tous les intervalles participent » **imposait** de passer à
`unit.intervals` : `intervals_t_extended[t]` filtre `b >= t`, donc supprimer les filtres
`t+1<=b` seuls aurait produit un `KeyError` sur les intervalles déjà terminés.

**Vérifié** : borne identique au baseline (1 817 736 à it. 10), donc les intervalles
auparavant exclus contribuent exactement 0 — cohérent avec $\tilde\tau = 0$ par récurrence
pour un intervalle terminé.

### A2 — Propagation $\tilde\tau$ pour les intervalles inactifs

**Statut** : **FAIT** · **Fichier** : `src/plugins/bellman_functions.jl:~536`

**Le piège de cette mise à jour.** Appliquer littéralement « Prop. 4 : pour $t \notin [a:b-1]$,
poser $\hat h = \hat\beta = 0$ » en sautant le bloc de calcul rend la borne **invalide** :
mesuré **1 905 374 > 1 850 053** (l'optimum exact) aux itérations 9–10.

Raison : l'éq. (36) est
$$\hat\tau_{j,a,b,t} = \hat h_{j,a,b,t} + \tilde\tau^\sigma_{j,a,b,t+1}.$$
Annuler $\hat h$ et $\hat\beta$ **n'annule pas** $\tilde\tau^\sigma$, la propagation des coupes
futures. Un intervalle qui démarre **plus tard** ($a > t+1$) est inactif au stade $t+1$ mais a
bien une valeur future non nulle. Sauter le bloc supprimait cette propagation.

La forme correcte :
```julia
active = a <= t+1 <= b-1
prop = sum(μ[i][k]*cut.intercept_gamma[unit.name,a,b] for (k,cut) in enumerate(improved_cuts))
if active
    intercept_gamma[...] += p * (Qjab + prop - π*yk)   # h_hat + tau~
    coefficients_gamma[...] += p * π                   # beta_hat
else
    intercept_gamma[...] += p * prop                   # h_hat = beta_hat = 0, tau~ conserve
end
```

> **Pour l'article** : la Prop. 4 et l'Algorithme 2 (lignes 19–20) disent « set
> $\hat h \leftarrow 0$ and $\hat\beta \leftarrow 0$ » sans rappeler que (36) ajoute ensuite
> $\tilde\tau^\sigma$. Lu vite, l'Algorithme 2 se code de la mauvaise façon. Une phrase
> explicite éviterait le piège.

Bug latent corrigé au passage : avant, un intervalle inactif recevait $\hat\beta = z \neq 0$,
multiplié dans la coupe par `power[j].out` — la puissance **physique** de l'unité, non nulle si
un **autre** intervalle sélectionné de la même unité est actif à ce stade. Inoffensif tant
qu'une unité n'a qu'un intervalle sélectionné, faux dès qu'elle en a deux.

---

## Écartés après analyse — ne pas « corriger »

### N1 — Point de linéarisation $\hat y^-$ (NON-BUG)

`incoming_state_j_a_b(unit, a, b, 0)` renvoie `InitialPower` pour **tous** les intervalles,
y compris ceux avec $a \ge 1$ où l'éq. (17) donne 0.

**Ce n'est pas une erreur.** Cette valeur sert de **point de linéarisation** $\hat y^-$ pour
choisir le multiplicateur local, pas d'état. Or $\hat Q(\cdot, z)$ est un max de deux fonctions
affines (éq. 48), donc **chacune des deux branches est une minorante affine valide partout** ;
$\hat y^-$ ne fait que sélectionner laquelle. La validité du support (35) ne dépend donc pas du
choix de $\hat y^-$ — même hors du domaine localement atteignable. Seule la **finesse** en dépend.

C'est le « any dual-feasible multiplier yields a valid coefficient » de la Prop. 4.

> **À faire côté article** : expliciter que les multiplicateurs sont choisis optimaux en un
> $\hat y^-$ de référence, et dire lequel. C'est une hypothèse de l'algorithme, pas une
> conséquence de (17).

### N2 — Domaine local vide (NON-BUG)

Conséquence de N1 : pour l'unité 2 (`P0 = 74,67`), intervalle `(a=5,b=15)` à `t=1`, on obtient
$[lo, hi] = [74{,}67,\ 0]$ — vide. `compute_Q_jab_MSUC` ne teste pas la vacuité.

**Sans conséquence sur la validité.** Ce qui entre dans la coupe n'est pas $Q$ mais l'ordonnée
à l'origine $h = Q - \pi\hat y^-$. Ici $Q = z \cdot 74{,}67$, $\pi = z$, donc $h = 0$ — le support
$(h,\beta) = (0, z)$ est valide. (Erreur d'audit initiale : j'avais regardé $Q$ au lieu de $h$.)

Un `@assert lo ≤ hi` reste souhaitable comme **documentation exécutable** de l'invariant, mais
ce n'est pas un correctif.

---

## À arbitrer

### B2b — État initial par intervalle dans la coupe racine

**Statut** : **FAIT** — tranché par la nouvelle §4.3, qui prescrit l'éq. (17) (« the initial
output prescribed by (17) »). Appliqué à la coupe racine et à `intercept_cstr`. · **Fichier** : `src/plugins/bellman_functions.jl:583`

À distinguer de [N1](#n1--point-de-linéarisation-ŷ--non-bug) : ici `unit.InitialPower` multiplie
`states_gamma` **à l'intérieur de l'expression de la coupe**, c'est donc l'**état**
$\tilde y_{j,a,b,0}$, pas un point de linéarisation.

```julia
node.states_gamma[unit.name, a, b] * (unit.InitialPower * coefficients_gamma[...] + intercept_gamma[...])
```

L'éq. (17) donne $\tilde y_{j,a,b,0} = y_{j,0}$ si $a \le 0$, **0** sinon.

**Testé** (patch `InitialPower * (a<=0)` appliqué aux deux endroits — coupe **et**
`intercept_cstr`, pour préserver la tension au point d'essai) : **aucune amélioration** de la
borne. Ce n'est donc pas un problème de validité en pratique.

**Décision à prendre côté article** : soit assumer $\tilde y_{j,a,b,0} := y_{j,0}$ pour tous les
intervalles et l'écrire, soit revenir à (17). À trancher en même temps que la formulation de
[N1](#n1--point-de-linéarisation-ŷ--non-bug), les deux relèvent de la même question :
*quel point porte l'état initial de chaque intervalle ?*

### R1 — Refonte du calcul de $\hat Q_{j,a,b,t}$

**Statut** : À ARBITRER

B1 a été possible parce que le calcul est éclaté en **4 fonctions parallèles**
(`compute_Pmin/Pmax/delta_up/delta_down_j_a_b`) qui refont chacune le découpage `t<a || t>=b`,
plus une 5ᵉ (`incoming_state_j_a_b`) qui est un copier-coller de `compute_Pmax_j_a_b` au cas
`t==0` près. `RU` et `RD` n'apparaissent jamais côte à côte, et les bornes $lo$/$hi$ de l'éq. (47)
ne sont jamais matérialisées.

**Couche A** — une seule source de vérité :
```julia
struct LocalRows          # les 4 coefficients de la ligne locale (43) pour (j,[a:b],t)
    p_lo :: Float64       # ỹ ≥ p_lo
    p_hi :: Float64       # ỹ ≤ p_hi   (cône démarrage ∧ Pmax ∧ cône arrêt)
    r_up :: Float64       # ỹ ≤ ỹ⁻ + r_up
    r_dn :: Float64       # ỹ ≥ ỹ⁻ − r_dn
end

function local_rows(unit, a, b, t)
    if t < a || t > b
        return LocalRows(0.0, 0.0, 0.0, 0.0)
    elseif t == b                                    # arrêt : ỹ = 0 forcé
        return LocalRows(0.0, 0.0, -unit.MinPower, unit.MinPower)
    else                                             # a ≤ t ≤ b-1
        return LocalRows(
            unit.MinPower,
            pmax_cone(unit, a, b, t),
            t == a ?  unit.MinPower : unit.DeltaRampUp,      # S^U = Pmin
            t == a ? -unit.MinPower : unit.DeltaRampDown,    # B1 : une seule occurrence
        )
    end
end
```

**Couche B** — le noyau pur (Prop. 6), testable sans `Node`, sans duals, sans solveur :
```julia
@inline function local_value(r::LocalRows, y⁻::Float64, z::Float64)
    lo = max(r.p_lo, y⁻ - r.r_dn)          # éq. (47)
    hi = min(r.p_hi, y⁻ + r.r_up)
    if z ≥ 0
        return y⁻ - r.r_dn ≥ r.p_lo ? (z*lo, -z*r.r_dn, z) : (z*lo, z*r.p_lo, 0.0)
    else
        return y⁻ + r.r_up ≤ r.p_hi ? (z*hi,  z*r.r_up, z) : (z*hi, z*r.p_hi, 0.0)
    end
end   # retourne (Q, h, β), invariant Q == h + β*y⁻
```

> **Piège à ne pas introduire en refactorant** : en `t == a`, le code a `delta_down = -MinPower`,
> soit $lo = \max(P_{\min}, \tilde y^- + P_{\min})$. Ce **n'est pas** le $R^D$ de l'éq. (47), mais
> c'est la ligne littérale de **(43c)**, et les deux coïncident quand $\tilde y^- = 0$. Ne pas
> « aligner sur (47) » en mettant `DeltaRampDown` partout — ce serait un nouveau bug.
> Le code est ici **plus fidèle que le papier**.

---

## P0 — Performance

### P1 — Coût en $O(k)$ par appel → entraînement en $O(k^2)$

**Statut** : TODO · **Fichiers** : `src/plugins/bellman_functions.jl:482` et `:558`

```julia
cost = unit.LinearTerm - ν[unit.Bus] +
       sum(μ[k]*cut.coefficients_gamma[unit.name, a, b] for (k,cut) in enumerate(node.improved_cuts))
```
soit **k lookups de hash** par appel. Mesuré sur l'instance jouet :

| | |
|---|---|
| appels à `compute_Q_jab_MSUC` / passe arrière | **398 618** |
| lookups Dict / passe arrière | **797 236 × k** |
| à k=100 coupes | **≈ 80 M lookups par passe arrière** |

Le même motif existe pour `intercept_gamma` dans `refine_bellman_function_extended`.
Coût linéaire en k ⇒ **entraînement en $O(k^2)$** : c'est ce qui rendra la campagne numérique
impraticable, indépendamment de toute question de correction.

**Correction** : $z_{j,a,b} = c_j - \nu_{n(j)} + \sum_k \mu_k \tilde\beta^k_{j,a,b}$ est un
**produit matrice-vecteur**. Indexer les intervalles de façon dense (`idx[j,a,b] ∈ 1:1968`,
calculé une fois au parsing) et remplacer dans `Cut2` les deux
`Dict{Tuple{Int,Int,Int},Float64}` par deux colonnes de matrices `n_int × k` :
```julia
z .= c .- ν[bus] .+ B[:, 1:k] * μ        # un gemv pour TOUS les intervalles
```
Gain mémoire au passage : 100 Dicts de 1968 entrées (~10 Mo fragmentés) → matrice contiguë de 1,5 Mo.

Bonus : `local_rows(j,a,b,t)` ne dépend que de données statiques ⇒ précalculable une fois
(47 232 entrées, ~1,5 Mo) au lieu de 1,6 M appels branchus par passe arrière.

⚠️ Touche `Cut2`, `refine_bellman_function_extended` **et** la réinstanciation des coupes dans
`forward_passes.jl:188-202` — le mécanisme le plus subtil du code.

---

## P1 — Bloquants pour produire des résultats

### B4 — Bloc `gamma` dupliqué au nœud 1

**Statut** : TODO · **Fichier** : `src/UC/builder.jl:289` (bloc à supprimer), cf. `:270`

Dans `subproblem_builder_UC_extended`, `gamma_var` **et** les 3 familles de contraintes de
liaison sont créées **deux fois** au nœud 1 : avant `if node == 1` (exécuté pour tous les nœuds),
puis à nouveau dans la branche `node == 1`. Le second bloc écrase les entrées du dictionnaire ;
le premier jeu reste dans le modèle, contraint à l'identique.

**Mesuré** : nœud 1 → **3936** variables `gamma_*` (attendu **1968**) ; nœud 2 → 1968 ✅

Pas faux mathématiquement, mais **double la taille du MIP racine**, résolu à chaque itération.

**Vérification** : `count(v -> startswith(name(v), "gamma_"), all_variables(m[1].subproblem)) == 1968`

### B5 — `parse_IEEE` ne s'exécute pas

**Statut** : TODO · **Fichier** : `src/UC/Struct/parsing.jl:125-130`

1. `set_intervals(...)` appelé avec **6 arguments**, seule méthode existante : **5**.
2. `for t in 1:T` → **`T` non défini** (c'est `TimeHorizon`).
3. `ThermalUnit(...)` construit avec **18 arguments** pour **20 champs** (manque `intervals_t_extended`).

Données pourtant présentes : `Data/IEEE/118_syst/` + `Data/IEEE/Scenarios/Wind.csv`.

> **La contribution 6 de l'article (étude IEEE 118-bus) est aujourd'hui inexécutable.**

**Correction** : aligner sur `parse_nc4` (lignes 33-35), qui est la version correcte.

### B6 — `calculate_bound` désactivée (`bound = 0.0`)

**Statut** : TODO · **Fichier** : `src/algorithm.jl:1145-1151`

```julia
# @_timeit_threadsafe model.timer_output "calculate_bound" begin ... end
bound = 0.0
```

Conséquences : `best bound : 0.000000e+00` dans tous les logs ; **toutes les règles d'arrêt
fondées sur la borne sont mortes** — la convergence finie de l'Algorithme 2 ne peut pas être
testée ; la vraie borne ne sort que par un `println` dans `src/plugins/forward_passes.jl:220`.

La fonction marche (appelée à la main sur le modèle 3-bin : 380 160).

**À vérifier en corrigeant** : `calculate_bound` passe par `solve_subproblem` (chemin 3-bin).
Pour `ExtendedFormulation`, s'assurer de la cohérence avec la réinstanciation des coupes au γ
courant, sinon écrire une variante `calculate_bound_extended`.

### B7 — Incompatibilité Julia 1.12 (MbedTLS ← HTTP)

**Statut** : TODO

`using MSUC` **échoue** sur Julia 1.12.6 (celui du `PATH`) : `MbedTLS` ne précompile pas.
`MbedTLS` vient de `HTTP`, utilisé **uniquement** par `src/visualization/dashboard.jl` (4 appels).
Le `Manifest.toml` est généré pour Julia **1.11.5** et `Pkg.status` le signale périmé.

**Correction** : retirer `HTTP` de `Project.toml` + le dashboard de `MSUC.jl`, puis `Pkg.resolve()`.
**Court terme** : documenter que Julia 1.11.5 est requis.

### E3 — Bruit non aligné entre les deux builders

**Statut** : TODO · **Fichier** : `src/UC/builder.jl`

- `subproblem_builder_UC_def` : `Ω = [[0.0 ...]]`, `P = [1.0]` → **déterministe**
- `subproblem_builder_UC_extended` : `S = 11`, `Ω = [[-0.05 + 0.1(s-1)/(S-1)]]`

Les deux builders **ne décrivent pas le même problème** : toute comparaison 3-bin ↔ extended
est faussée en l'état.

**Correction** : sortir la définition du bruit des builders (paramètre de l'instance ou argument).
Prérequis de [S1](#s1--aucun-test).

### S1 — Aucun test

**Statut** : TODO — pas de `test/`, pas de `runtests.jl`.

Par ordre d'utilité :
1. **Validation déterministe** (cf. [protocole](#protocole-de-validation--le-cas-déterministe)) :
   `force=0`, LB croissante, LB ≤ optimum exact, LB → optimum. C'est le test qui tranche.
2. **Unitaires sur les closed-forms** : `compute_delta_{up,down}_j_a_b`, `compute_P{min,max}_j_a_b`
   — c'est là qu'était B1. Idéalement : comparer `local_value` à un LP JuMP 1-D sur une grille
   de $(\tilde y^-, z)$.
3. **Invariant du support** : `Q == h + β*ỹ⁻` (attrape les erreurs de signe).
4. **Tension au point d'essai** (Prop. 4) : égalité à `tol` près.
5. **Équivalence Prop. 3** : 3-bin et extended convergent vers la même valeur
   (prérequis : [E3](#e3--bruit-non-aligné-entre-les-deux-builders)) — attention, ne vaut
   qu'**à convergence** des deux (cf. piège d'interprétation ci-dessus).

---

## P2 — Écarts article ↔ code

### E1 — Le processus ARMA ζ n'est pas implémenté

**Statut** : TODO

Aucune occurrence de `ζ`, `Ψ`, `E` dans `src/`. L'article définit l'état $(x, y_{t-1}, \zeta_{t-1})$,
la récursion (1g) $\zeta_t = \Psi\zeta_{t-1} + E\xi_t$, et les coefficients
$\delta_{t,\xi} = \Psi^\top\hat\lambda_\xi$ en (12)/(37). Le code utilise un bruit i.i.d. codé en
dur dans le builder (11 valeurs, ±5 %, une seule ferme). **Tout le volet $\delta/\lambda$ est absent.**

**Décision** : implémenter l'ARMA, ou restreindre explicitement l'article au cas
stagewise-independent. À trancher avant la campagne numérique — cela change l'espace d'état.

### E2 — Lignes de shutdown-feasibility (44)–(46)

**Statut** : TODO

Le builder contient une forme agrégée correspondant à (46), mais **les lignes 3-bin (45) par
décalage $k$ ne sont pas implémentées**. Sans elles, l'Hypothèse 3 peut échouer côté 3-bin
→ sous-problème infaisable.

> Indice : `model_infeasible_node_24.cuts.json` et `subproblem_24.mof.json` sont des artefacts
> de débogage d'une **infaisabilité au nœud 24**. Lien plausible, **à confirmer**.

### S4 — Runs stochastiques non reproductibles

**Statut** : TODO

La passe avant échantillonne $\xi_t$ sans graine fixée : deux exécutions identiques donnent
des bornes différentes (observé : 685 151, 710 541, 927 852 à l'itération 5 sur la même
configuration). Impossible de comparer deux variantes de code en stochastique, ni de
reproduire une table de l'article.

> C'est précisément pourquoi le [protocole de validation](#protocole-de-validation--le-cas-déterministe)
> utilise `force = 0` : les 11 aléas deviennent identiques, l'échantillonnage n'a plus d'effet,
> et les bornes sont reproductibles au bit près. À garder pour toute comparaison A/B.

**Correction** : exposer une graine (`Random.seed!` ou un RNG passé à `train`).

### S2 — Horizon `T=24` codé en dur

**Statut** : TODO

`bmax=25`, `b <= 24`, `min(b,24)`, `power[1:24]` : ~22 occurrences.
Bloque le paramètre `T` de `parse_nc4` et surtout **empêche de construire une petite instance**
(`T=4`, 2 unités) dont l'équivalent déterministe **stochastique** serait calculable — ce qui
étendrait le protocole de validation au cas avec bruit. Fort effet de levier sur [S1](#s1--aucun-test).

### S3 — Nettoyage du dépôt

**Statut** : TODO

- [ ] `.gitignore` (absent) + sortir du suivi : `MSUC.log` (645 Ko, **append-only à chaque `train`**),
      `subproblem_24.mof.json` (2 Mo), `model_infeasible_node_24.cuts.json`
      (⚠️ archiver les deux derniers : seules traces de l'infaisabilité nœud 24, cf. [E2](#e2--lignes-de-shutdown-feasibility-4446))
- [ ] Supprimer les ~130 lignes commentées de `subproblem_builder_UC_extended` (`builder.jl:106-236`,
      version avec les `z_{j,a,b}`) — historique dans le commit `b043fdd`
- [ ] Supprimer les blocs commentés de `refine_bellman_function_extended` (`bellman_functions.jl:596-645`)
- [ ] `README.md` : **Julia 1.11.5 requis**, licence Gurobi, commande de reproduction
- [ ] `Pkg.resolve()` (`Manifest.toml` périmé vs `Project.toml`)
- [ ] `toy_1d.ipynb` : notebook de brouillon (cellules 16-18 = test JuMP sans rapport)

---

## Ce qui est correct — à ne pas « corriger »

- **Le closed-form de la Proposition 6 est juste.** `compute_Q_jab_MSUC` implémente exactement
  (48) et (49), branche par branche, y compris $h = Q - \pi\tilde y^-$.
- **L'astuce `Cut2` + réinstanciation est correcte et élégante.** Plutôt que de porter les
  produits $z_{j,a,b} = \gamma\cdot y$ (commit `b043fdd` « trop de variables z »), le code stocke
  les coefficients par intervalle et **réécrit la ligne** via `set_normalized_rhs` /
  `set_normalized_coefficient` au γ courant (`forward_passes.jl:188-202`). C'est exactement (41g)
  à planning fixé.
- **La coupe racine est bien bilinéaire en γ** (`node.states_gamma`) : c'est elle qui porte la
  force supplémentaire de la formulation intervalles.
- **Le cône de démarrage dans `compute_Pmax_j_a_b`** était un renforcement **licite** (valide sur
  le domaine localement atteignable $\mathfrak{Y}_{j,a,b,t}$), pas un bug — retiré depuis par
  fidélité à (46), sans effet mesurable (cf. [A1](#a1--alignement-sur-la-nouvelle-section-4)).

### Invariant à ne pas casser : $\hat\alpha$ est rétro-calculé

`intercept_cstr` ne calcule **pas** $\hat\alpha$ par l'éq. (37). Il le **rétro-calcule** depuis
`intercept` (la valeur de Bellman au point d'essai), de sorte que la coupe soit tendue par
construction :
$$\hat\alpha = \texttt{intercept} - \sum \hat\gamma_{j,a,b}\bigl(\hat\tau_{j,a,b} + \hat\beta_{j,a,b}\,\tilde y_{j,a,b}\bigr)$$

Ce rétro-calcul **suppose la dualité forte**, donc des multiplicateurs **optimaux à l'état
d'essai** sur les intervalles sélectionnés — exactement l'exigence de la Prop. 4
(« require this multiplier to be optimal in (32) whenever $\hat\gamma_{j,a,b} = 1$ »).

Conséquence pratique : sur un intervalle **sélectionné**, le point de linéarisation `yk` doit
être l'état réel, pas la trajectoire de référence. Linéariser ailleurs surestimerait
$\hat\alpha$ et rendrait la coupe **invalide**. Sur un intervalle non sélectionné, `yk` est
libre ($q$ de l'éq. (49)) puisque $\hat\gamma \approx 0$ annule le terme.

> À distinguer de [N1](#n1--point-de-linéarisation-ŷ--non-bug) : « le point de linéarisation est
> libre » vaut pour la **validité du support local**, pas pour le rétro-calcul de $\hat\alpha$
> sur un intervalle sélectionné.

---

## Reproduction

```bash
# ⚠️ Julia 1.11.5, pas celui du PATH (1.12.6) — cf. B7
JL=/home/mathis/julia-1.11.5/bin/julia
$JL --project=. -e 'using MSUC; println("LOAD_OK")'
```

La vraie borne inférieure est le **premier** nombre de `Lower bound: <LB>, <coût d'engagement>`
(le second est le coût d'engagement seul ; les deux coïncident tant que θ = 0).

---

## Journal

| Date | ID | Action |
|------|----|--------|
| 2026-09-13 | — | Audit initial |
| 2026-09-13 | — | Mise en place du protocole de validation déterministe (optimum exact = 1 850 053,2) |
| 2026-09-13 | B1 | **Corrigé** `DeltaRampUp` → `DeltaRampDown`. Borne it.10 : 1 765 926 → 1 817 736 (95,5 % → 98,3 %) |
| 2026-09-13 | N1, N2 | Classés **NON-BUG** : $\hat y^-$ est un point de linéarisation, toute branche de (49) est une minorante valide |
| 2026-09-13 | — | **Rétractation** : la narration « coupes invalides » de l'audit initial était fausse. Le générateur étendu est valide (test déterministe) |
| 2026-09-13 | A1 | Alignement sur la §4 réécrite : $\bar p$ (46), $q$ (49), (17), intervalles actifs, tous les intervalles participent |
| 2026-09-13 | A2 | Propagation $\tilde\tau^\sigma$ conservée pour les intervalles inactifs — sans quoi la borne dépasse l'optimum (1 905 374 > 1 850 053) |
| 2026-09-13 | — | `refine_bellman_function_extended` nettoyée : 150 → 108 lignes, code mort supprimé, `mubar` hoisté (parcours des coupes divisé par `nscen`). Bornes inchangées |
| 2026-09-13 | NOT | **Refactor notation** : `μ`→`σ` (c'était le σ de l'article, pas le μ), `Cut2`→`IntervalCut(alpha, tau, beta)`, `improved_cuts`→`interval_cuts`, helpers §4 renommés (`p_lower`, `p_upper`, `ramp_up`, `ramp_down`, `reference_output`, `initial_interval_output`, `local_support`) |
| 2026-09-13 | R1 | `local_support` remplace `compute_Q_jab_MSUC` : renvoie $(\hat h,\hat\beta)$ directement. Supprime la soustraction catastrophique $z(y-R^D)-zy$ → coefficients exacts, borne it.10 : 1 817 736 → **1 820 387** |
| 2026-09-13 | B4,B5,B6,B7 | γ dupliqué supprimé ; `parse_IEEE` réparé ; borne réelle dans le journal ; HTTP+dashboard retirés → **Julia 1.12 fonctionne** |
| 2026-09-13 | E2,E3 | Lignes (44) implémentées côté 3-bin (plus de variables γ dans le builder 3-bin) ; bruit partagé via `forecast_error_support` |
| 2026-09-14 | S5 | `write_cuts_to_file` / `read_cuts_from_file` (persistance de politique SDDP.jl) retirées, avec `_node_name_parser` et le dump JSON de diagnostic d'`attempt_numerical_recovery`. `read_` n'était jamais appelée ; `write_` ne sérialisait que les coupes génériques (`global_theta`, `local_thetas`) et **pas** les `IntervalCut`, donc son dump était vide en formulation étendue — c'est ce qu'on voyait dans `model_infeasible_node_24.cuts.json`. Le `subproblem_*.mof.json`, lui, est conservé : il est rejouable dans le solveur. Dépendance `JSON` devenue inutile, retirée |
| 2026-09-14 | S5 | Handlers de dualité SDDiP retirés : `LagrangianDuality`, `StrengthenedConicDuality`, `BanditDuality`, `FixedDiscreteDuality`, plus `plugins/local_improvement_search.jl` (BFGS / approximation extérieure, utilisé seulement par la dualité lagrangienne) et les shims de migration SDDP.jl v0.4.0. Sans objet ici : §1 du papier — seule la racine est non convexe, chaque étage suivant est un LP, donc `ContinuousConicDuality` est exacte. `duality_handlers.jl` passe de 740 à 135 lignes |
| 2026-09-14 | S5 | **845 lignes nettes supprimées** : trois fonctionnalités SDDP.jl inatteignables avec `LinearGraph`. (1) `belief_state` — graphes partiellement observables ; `belief_partition` n'est rempli que par `add_ambiguity_set`, donc `node.belief_state` valait toujours `nothing`. (2) `objective_state` — objectifs dépendant d'un prix exogène ; positionné seulement par `add_objective_state`, jamais appelé. (3) code de cycle (`terminated_due_to_cycle`, `starting_states`, `cycle_discretization_delta`, `distance`, `inf_norm`) — un chemin ne boucle pas. Bugs latents corrigés au passage : `new{T}` de `PolicyGraph` et `Graph(root_node)` passaient des arguments surnuméraires après retrait des champs, ce que la précompilation n'attrape pas |
| 2026-09-14 | S1 | `multistage_extensive` (équivalent déterministe de l'arbre) et `simulate_policy` (majorant par simulation exhaustive). À T=4, S=2 : borne = optimum = politique = 235 205,1131 pour les **deux** formulations, en 6 et 15 itérations. Valide l'ARMA, l'état $\zeta$ et $\tilde\delta$ |
| 2026-09-14 | B9 | **La borne publiée n'était pas valide** : `objective_value` du MIP racine renvoie l'incumbent, qui majore l'optimum de la tolérance `MIPGap`. Remplacé par `objective_bound` (borne duale) via `root_outer_bound`, et publication du **maximum courant** des bornes duales — celles-ci sont valides mais non monotones. Corrige aussi les reculs de borne observés à $\sigma = 3$ |
| 2026-09-14 | E1 | **Processus ARMA implémenté** : ARMA(p,q) sous forme d'état (1g) avec les paramètres PSCC (40), $\zeta$ comme variable d'état, et le coefficient de coupe $\tilde\delta_t = \Psi^\top(S_\eta^\top \hat D_t \bar\nu + \tilde\delta^\sigma_{t+1})$ des éq. (10)/(12)/(40b). Discrétisation à support fini paramétrable (Gauss-Hermite ou équiprobable, variance renormalisée). Vérifié : récursion conforme à (40) à 1e-18, $\delta^\xi/\delta^\eta = \Theta_1/\Phi_1$, non-régression exacte à `force = 0` |
| 2026-09-14 | B8 | **Convention $b = T+1$ (Sect. 3.1) absente** de `set_intervals` : les intervalles $[a, T{+}1]$ ne respectant pas la durée minimale de marche étaient exclus. L'ensemble admissible du premier étage était donc **plus petit** que celui de la 3-bin, cassant la Prop. 3 (57 intervalles manquants à T=24 ; root infaisable à T=4). `set_intervals_t` dérive maintenant de `set_intervals` par filtrage, ce qui garantit $T_j(t) \subseteq T_j$ |
| 2026-09-14 | — | Two-stage Benders retiré (7 fichiers + `tools.jl`) ; `extensive_formulation.jl` conservée et déplacée en `src/UC/` ; champs morts `model_Q_jab`, `Tup`, `Tdown` |
| 2026-09-14 | S1 | `examples/compare_formulations.jl` et `examples/toy_convergence.jl` : à T=4 les deux formulations convergent vers 233 802,50 en 8 et 14 itérations |
| 2026-09-13 | S3 | `states_z`, `intervals_t_extended`, builder commenté (~130 l.), blocs de debug supprimés ; `builder.jl` factorisé ; **README.md** écrit |
