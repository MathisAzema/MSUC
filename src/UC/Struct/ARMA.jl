# =============================================================================
#  Processus ARMA(p,q) de l'erreur de prevision de demande nette (Sect. 2.1).
#
#  Notation du papier MSUC :
#     eta_t = sum_i Phi_i eta_{t-i} + xi_t + sum_k Theta_k xi_{t-k}
#  avec eta_t l'erreur RELATIVE par noeud et xi_t l'innovation, a support fini
#  (Hypothese 1). Le processus devient markovien apres augmentation de l'etat :
#     zeta_t = (eta_t, .., eta_{t-p+1}, xi_t, .., xi_{t-q+1})  in R^{(p+q)|N|}
#     zeta_t = Psi zeta_{t-1} + E xi_t                          (1g)
#     eta_t  = S_eta zeta_t
#
#  ATTENTION aux notations croisees avec l'article PSCC, eq. (40), d'ou viennent
#  les parametres calibres : ce que PSCC note eta (erreur) est notre eta, mais ce
#  que PSCC note eta_t ~ N(0, .) est notre INNOVATION xi_t. La correspondance est
#     PSCC epsilon <-> MSUC eta   (erreur relative)
#     PSCC eta     <-> MSUC xi    (innovation)
# =============================================================================

"Coefficients AR de l'ARMA(1,1) calibre sur la demande francaise, PSCC eq. (40)."
const ARMA_PHI = [0.67]

"Coefficients MA de l'ARMA(1,1) calibre, PSCC eq. (40)."
const ARMA_THETA = [0.117]

"""
Variance de l'innovation, PSCC eq. (40) : `xi_t ~ N(0, 2.9e-4 * sigma)`.

Le second argument de `N(.,.)` est lu comme une VARIANCE : pour sigma = 1 l'ecart-type
de l'innovation vaut sqrt(2.9e-4) ~ 1.7 %, ordre de grandeur d'une erreur de prevision
de demande a un jour. `sigma` echelonne la variance (donc l'ecart-type en sqrt(sigma)).
"""
const ARMA_INNOVATION_VARIANCE = 2.9e-4

"""Taille par defaut du support de l'innovation xi_t (Hypothese 1)."""
const ARMA_SUPPORT_SIZE = 5

"""Schema de discretisation par defaut : `:gauss_hermite` ou `:equiprobable`."""
const ARMA_DISCRETIZATION = :gauss_hermite

"""
    gauss_hermite(S) -> (noeuds, poids)

Quadrature de Gauss-Hermite a `S` points pour la loi N(0,1), par Golub-Welsch : les
noeuds sont les valeurs propres de la matrice de Jacobi de la recurrence des polynomes
d'Hermite probabilistes, les poids les carres des premieres composantes des vecteurs
propres. Exacte pour les polynomes de degre <= 2S-1, donc les deux premiers moments
de la loi discretisee sont exacts.
"""
function gauss_hermite(S::Int)
    S <= 1 && return ([0.0], [1.0])
    J = LinearAlgebra.SymTridiagonal(zeros(S), [sqrt(k) for k in 1:(S-1)])
    F = LinearAlgebra.eigen(J)
    return collect(F.values), [abs2(F.vectors[1, i]) for i in 1:S]
end

"""
    standard_normal_quantile(p)

Quantile de N(0,1) par l'approximation rationnelle d'Acklam (erreur relative < 1.2e-9).
Evite une dependance a SpecialFunctions pour le seul schema equiprobable.
"""
function standard_normal_quantile(p::Float64)
    @assert 0 < p < 1
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= phigh
        q = p - 0.5; r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6]) * q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

"""
    equiprobable_normal(S) -> (noeuds, poids)

Discretisation de N(0,1) en `S` strates EQUIPROBABLES : chaque scenario a la probabilite
1/S et pour noeud la moyenne conditionnelle de sa strate,
    m_i = S (phi(z_{i-1}) - phi(z_i)),   z_i = Phi^{-1}(i/S),
ou phi est la densite standard. Preserve exactement la moyenne, mais pas les moments
d'ordre superieur : la variance est sous-estimee (effet de regroupement intra-strate).

A preferer a Gauss-Hermite si l'on veut des scenarios de poids egal ou une meilleure
representation des queues, par exemple pour une mesure de risque type AVaR.
"""
function equiprobable_normal(S::Int)
    S <= 1 && return ([0.0], [1.0])
    φ(z) = exp(-z^2 / 2) / sqrt(2π)
    z = [standard_normal_quantile(i / S) for i in 1:(S-1)]
    dens = vcat(0.0, φ.(z), 0.0)
    return [S * (dens[i] - dens[i+1]) for i in 1:S], fill(1 / S, S)
end

"""
    innovation_support(S, sd; discretization, match_variance) -> (noeuds, poids)

Support fini de l'innovation d'ecart-type `sd`.

* `:gauss_hermite` : les 2S-1 premiers moments sont exacts. Optimal pour une esperance,
  mais les noeuds extremes portent des probabilites tres faibles (8.6e-10 a S=15), donc
  les queues sont peu representees en pratique.
* `:equiprobable` : tous les scenarios ont le poids 1/S et les noeuds restent proches du
  centre. A preferer pour une mesure de risque type AVaR.

`match_variance` renormalise les noeuds pour que la variance discrete vaille exactement
`sd^2`. Sans cela le schema equiprobable la SOUS-ESTIME nettement (0.897 sd^2 a S=5,
0.976 a S=15), la dispersion intra-strate etant perdue. Sans effet sur Gauss-Hermite,
dont la variance est deja exacte.
"""
function innovation_support(
    S::Int,
    sd::Float64;
    discretization::Symbol = ARMA_DISCRETIZATION,
    match_variance::Bool = true,
)
    nodes, weights = if discretization === :gauss_hermite
        gauss_hermite(S)
    elseif discretization === :equiprobable
        equiprobable_normal(S)
    else
        error("discretization inconnue : $discretization (:gauss_hermite ou :equiprobable)")
    end
    if match_variance && S > 1
        m2 = sum(weights .* nodes .^ 2)
        m2 > 0 && (nodes ./= sqrt(m2))
    end
    return sd .* nodes, weights
end

"""
    ARMAProcess

Processus ARMA(p,q) de l'erreur relative de prevision, sous la forme d'etat (1g).

* `Phi`, `Theta` : coefficients AR et MA, appliques independamment a chaque noeud.
* `Psi`, `E`, `S_eta` : matrices de (1g) et de l'extraction `eta_t = S_eta zeta_t`.
* `support`, `probability` : discretisation a support fini de `xi_t` (Hypothese 1),
  une realisation est un vecteur de longueur |N|.
* `zeta0` : etat initial, donnee du probleme.
"""
struct ARMAProcess
    Phi::Vector{Float64}
    Theta::Vector{Float64}
    Psi::Matrix{Float64}
    E::Matrix{Float64}
    S_eta::Matrix{Float64}
    support::Vector{Vector{Float64}}
    probability::Vector{Float64}
    zeta0::Vector{Float64}
end

"""
    ARMAProcess(nbus; sigma, Phi, Theta, S, common_innovation)

Construit le processus et ses matrices compagnons pour `nbus` noeuds.

`sigma` est le parametre d'echelle de PSCC eq. (40) : la variance de l'innovation vaut
`2.9e-4 * sigma`. `sigma = 1` correspond a l'erreur de prevision de demande seule, des
valeurs plus grandes a l'incertitude additionnelle de la production eolienne.

`S` est la taille du support de l'innovation et `discretization` son schema
(`:gauss_hermite` ou `:equiprobable`). Avec `common_innovation = true` une seule
innovation scalaire est tiree et appliquee a tous les noeuds, ce qui garde le nombre
d'aleas par etape egal a `S` ; sinon il faudrait `S^|N|` combinaisons, impraticable.
"""
function ARMAProcess(
    nbus::Int;
    sigma::Float64 = 1.0,
    Phi::Vector{Float64} = ARMA_PHI,
    Theta::Vector{Float64} = ARMA_THETA,
    variance::Float64 = ARMA_INNOVATION_VARIANCE * sigma,
    S::Int = ARMA_SUPPORT_SIZE,
    discretization::Symbol = ARMA_DISCRETIZATION,
    match_variance::Bool = true,
    common_innovation::Bool = true,
)
    p, q = length(Phi), length(Theta)
    n = (p + q) * nbus
    blk(i) = ((i-1)*nbus+1):(i*nbus)          # bloc i de zeta
    Id = Matrix{Float64}(LinearAlgebra.I, nbus, nbus)

    Psi = zeros(n, n)
    # Ligne bloc 1 : eta_t = sum Phi_i eta_{t-i} + sum Theta_k xi_{t-k}  (+ xi_t via E)
    for i in 1:p
        Psi[blk(1), blk(i)] .= Phi[i] .* Id
    end
    for k in 1:q
        Psi[blk(1), blk(p+k)] .= Theta[k] .* Id
    end
    # Lignes blocs 2..p et p+2..p+q : decalages des retards
    for i in 2:p
        Psi[blk(i), blk(i-1)] .= Id
    end
    for k in 2:q
        Psi[blk(p+k), blk(p+k-1)] .= Id
    end
    # E : l'innovation entre dans eta_t et est stockee dans le bloc p+1
    E = zeros(n, nbus)
    E[blk(1), :] .= Id
    q >= 1 && (E[blk(p+1), :] .= Id)

    S_eta = zeros(nbus, n)
    S_eta[:, blk(1)] .= Id

    nodes, weights = innovation_support(
        S,
        sqrt(variance);
        discretization = discretization,
        match_variance = match_variance,
    )
    support = if common_innovation
        [fill(z, nbus) for z in nodes]
    else
        nbus == 1 ? [[z] for z in nodes] :
        error("common_innovation = false demanderait S^|N| aleas par etape")
    end
    return ARMAProcess(Phi, Theta, Psi, E, S_eta, support, weights, zeros(n))
end

"Dimension de l'etat d'incertitude zeta_t."
zeta_dimension(a::ARMAProcess) = size(a.Psi, 1)
