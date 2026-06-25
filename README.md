# Gestion des Risques — Analyse Financière de Renault Retail Group (RRG)

> **Auteur :** Yasser Houssein Hassan
> **Contexte :** Master 2 — Ingénierie Mathématique en Finance et Logistique (IMFL)
> **Domaine :** Finance d'entreprise, analyse financière, quantification du risque, cartographie des risques.

Ce projet présente un **audit complet de la gestion des risques** de **Renault Retail Group (RRG)**, filiale de distribution automobile du groupe Renault. Il combine une **analyse financière** rigoureuse (bilan, compte de résultat, ratios) à une **quantification statistique du risque** (Value-at-Risk, Expected Shortfall) et à une **cartographie matricielle** des risques d'activité.

---

## 1. Analyse financière de RRG

### 1.1 Évolution de l'activité (2016–2023)

Le chiffre d'affaires recule nettement après 2019 sous l'effet conjugué de :
- la **crise sanitaire** (rupture de la demande et des approvisionnements) ;
- la **transition technologique** (passage du thermique à l'électrique) ;
- la **concurrence des acteurs digitaux** de la distribution automobile.

### 1.2 Diagnostic 2023 (en millions d'euros)

| Poste | Valeur | Lecture |
|---|---|---|
| Résultat net | **−3,74 M€** | Déficit sous pression des investissements de transition |
| Dette nette | **≈ 1 200 M€** | Endettement financier élevé |
| Capitaux propres | **25,4 M€** | Marge de manœuvre très réduite |

La structure financière révèle un **levier d'endettement (*gearing*) très élevé** et une **faible autonomie financière**, exposant l'entreprise au risque de solvabilité et à la hausse des taux d'intérêt.

### 1.3 Ratios d'analyse financière implémentés

| Ratio | Formule | Interprétation |
|---|---|---|
| Liquidité générale | $\dfrac{\text{Actif circulant}}{\text{Passif circulant}}$ | Capacité à honorer le court terme ($>1$) |
| Liquidité réduite | $\dfrac{\text{AC} - \text{Stocks}}{\text{Passif circulant}}$ | Test acide (hors stocks) |
| BFR | $\text{AC} - \text{PC}$ | Besoin en fonds de roulement |
| Gearing | $\dfrac{\text{Dette nette}}{\text{Capitaux propres}}$ | Levier financier |
| Autonomie financière | $\dfrac{\text{Capitaux propres}}{\text{Total actif}}$ | Indépendance vis-à-vis des créanciers |
| Dette nette / EBITDA | $\dfrac{\text{Dette nette}}{\text{EBITDA}}$ | Soutenabilité de la dette ($<3\text{–}4$) |

---

## 2. Quantification du risque de marché

La valeur en risque (*Value-at-Risk*) mesure la perte potentielle d'un portefeuille à un horizon et un niveau de confiance $\alpha$ donnés.

### 2.1 Value-at-Risk (VaR)

- **VaR historique** — quantile empirique des rendements :

$$\text{VaR}_\alpha = -\,q_{1-\alpha}(r).$$

- **VaR paramétrique gaussienne** :

$$\text{VaR}_\alpha = -\left(\mu + z_{1-\alpha}\,\sigma\right), \qquad z_{1-\alpha} = \Phi^{-1}(1-\alpha).$$

- **VaR Monte-Carlo** — par simulation d'un grand nombre de scénarios.

### 2.2 Expected Shortfall (CVaR)

La VaR ne renseigne pas sur l'ampleur des pertes **au-delà** du seuil. L'*Expected Shortfall* (mesure de risque **cohérente** au sens d'Artzner et al., 1999, car sous-additive) corrige ce défaut :

$$\text{ES}_\alpha = -\,\mathbb{E}\!\left[\,r \mid r \leq -\text{VaR}_\alpha\,\right].$$

---

## 3. Cartographie et évaluation des risques

Chaque risque est évalué par sa **criticité** :

$$\text{Criticité} = \text{Probabilité} \times \text{Impact},$$

puis classé en niveaux *Faible / Modéré / Élevé / Critique*. Les familles de risques identifiées :

| Famille | Exemples | Probabilité |
|---|---|---|
| **Financier** | Solvabilité, BFR, hausse des taux | 65–90 % |
| **Opérationnel** | Chaîne d'approvisionnement, RH | moyenne |
| **Juridique & environnemental** | Pénalités CO2 (norme CAFE) | élevée |
| **Réputationnel** | Scandales produits, communication | faible-moyenne |

---

## 4. Présentation du code Python

Les outils quantitatifs sont implémentés dans **`analyse_risque.py`** (`numpy`, `pandas`, `scipy`).

### 4.1 Ratios de structure financière

```python
def ratios_financiers(bilan, compte_resultat):
    return {
        "ratio_liquidite_generale": bilan["actif_circulant"] / bilan["passif_circulant"],
        "gearing": bilan["dette_nette"] / bilan["capitaux_propres"],
        "autonomie_financiere": bilan["capitaux_propres"] / bilan["total_actif"],
        "dette_nette_sur_ebitda": bilan["dette_nette"] / compte_resultat["ebitda"],
        "marge_nette": compte_resultat["resultat_net"] / compte_resultat["chiffre_affaires"],
    }
```

### 4.2 Value-at-Risk et Expected Shortfall

```python
from scipy import stats

def var_parametrique(rendements, alpha=0.95):
    mu, sigma = np.mean(rendements), np.std(rendements, ddof=1)
    z = stats.norm.ppf(1 - alpha)        # quantile gaussien
    return -(mu + z * sigma)

def expected_shortfall(rendements, alpha=0.95):
    seuil = np.quantile(rendements, 1 - alpha)
    queue = rendements[rendements <= seuil]   # pertes au-delà de la VaR
    return -queue.mean()
```

### 4.3 Cartographie des risques

```python
def cartographie_risques(risques):
    df = risques.copy()
    df["criticite"] = df["probabilite"] * df["impact"]   # criticité
    df["niveau"] = df["criticite"].apply(_classer)        # Faible -> Critique
    return df.sort_values("criticite", ascending=False)
```

### 4.4 Exécution

```bash
pip install numpy pandas scipy
python analyse_risque.py
```

Le script calcule les ratios financiers de RRG (exercice 2023), estime la VaR (historique, paramétrique, Monte-Carlo) et l'Expected Shortfall aux seuils 95 % et 99 %, et produit la cartographie des risques classée par criticité décroissante.

---

## 5. Stratégies et recommandations

1. **Diversification & innovation** — mix de 40 % de véhicules électriques/hybrides d'ici 2027, partenariats batteries (Verkor), services de mobilité partagée (« RRG Mobility »).
2. **Gestion financière proactive** — couverture de la dette à taux fixe, optimisation du BFR, *cash pooling* européen.
3. **Optimisation des opérations** — logistique intelligente (IoT, traçabilité blockchain), modèles probabilistes de gestion des stocks (*Just-in-Time*).
4. **Conformité & cybersécurité** — gouvernance RGPD/CSRD, SOC dédié, chiffrement AES-256.

---

## 6. Structure du dépôt

| Fichier | Description |
|---|---|
| `analyse_risque.py` | Outils Python : ratios financiers, VaR/CVaR, cartographie. |
| `gestion_de_risque.pdf` | Rapport final (15 pages) : bilans, graphiques, matrice des risques. |
| `README.md` | Le présent document. |

---

## Références

- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). *Coherent Measures of Risk*. Mathematical Finance.
- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
- McNeil, A., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management*. Princeton University Press.

---
*Projet réalisé par Yasser Houssein Hassan*
