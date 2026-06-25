# -*- coding: utf-8 -*-
"""
============================================================================
Gestion des Risques — Analyse financière & quantification (Renault RRG)
============================================================================

Outils quantitatifs d'appui à l'audit de gestion des risques de Renault
Retail Group (RRG) :
    1. Indicateurs de structure financière (solvabilité, liquidité, BFR,
       gearing) à partir du bilan et du compte de résultat.
    2. Mesures de risque de marché : Value-at-Risk (VaR) et Expected
       Shortfall (CVaR), historique, paramétrique gaussienne, Monte-Carlo.
    3. Cartographie des risques : criticité = probabilité × impact.

Auteur : Yasser Houssein Hassan
Contexte : Master 2 Ingénierie Mathématique en Finance et Logistique (IMFL)
Dépendances : numpy, pandas, scipy
============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Ratios de structure financière
# ---------------------------------------------------------------------------
def ratios_financiers(bilan: dict, compte_resultat: dict) -> dict:
    r"""Indicateurs clés d'analyse financière.

    bilan : actif_circulant, passif_circulant, stocks, dette_nette,
            capitaux_propres, total_actif.
    compte_resultat : chiffre_affaires, resultat_net, ebitda, charges_int.
    """
    ac = bilan["actif_circulant"]
    pc = bilan["passif_circulant"]
    cp = bilan["capitaux_propres"]
    dn = bilan["dette_nette"]
    ebitda = compte_resultat["ebitda"]

    return {
        # Liquidité générale : actif circulant / passif circulant (> 1 souhaité)
        "ratio_liquidite_generale": ac / pc,
        # Liquidité réduite (acid test) : hors stocks
        "ratio_liquidite_reduite": (ac - bilan["stocks"]) / pc,
        # Besoin en fonds de roulement
        "BFR": ac - pc,
        # Gearing (levier) : dette nette / capitaux propres
        "gearing": dn / cp,
        # Autonomie financière : capitaux propres / total actif
        "autonomie_financiere": cp / bilan["total_actif"],
        # Levier d'endettement : dette nette / EBITDA (< 3-4 souhaité)
        "dette_nette_sur_ebitda": dn / ebitda if ebitda else np.inf,
        # Marge nette
        "marge_nette": compte_resultat["resultat_net"] / compte_resultat["chiffre_affaires"],
    }


# ---------------------------------------------------------------------------
# 2. Mesures de risque de marché : VaR et Expected Shortfall
# ---------------------------------------------------------------------------
def var_historique(rendements: np.ndarray, alpha: float = 0.95) -> float:
    r"""Value-at-Risk historique au niveau de confiance alpha.

    VaR_α = - quantile_{1-α}(rendements). Perte maximale non dépassée avec
    probabilité α sur l'horizon considéré.
    """
    return -np.quantile(rendements, 1 - alpha)


def var_parametrique(rendements: np.ndarray, alpha: float = 0.95) -> float:
    r"""VaR gaussienne :  VaR_α = -(μ + z_{1-α} σ),  z_{1-α} = Φ⁻¹(1-α)."""
    mu, sigma = np.mean(rendements), np.std(rendements, ddof=1)
    z = stats.norm.ppf(1 - alpha)
    return -(mu + z * sigma)


def expected_shortfall(rendements: np.ndarray, alpha: float = 0.95) -> float:
    r"""Expected Shortfall (CVaR) : perte moyenne au-delà de la VaR.

    ES_α = -E[r | r ≤ -VaR_α]. Mesure cohérente du risque (sous-additivité).
    """
    seuil = np.quantile(rendements, 1 - alpha)
    queue = rendements[rendements <= seuil]
    return -queue.mean() if queue.size else np.nan


def var_monte_carlo(mu, sigma, alpha=0.95, n_sim=100_000, graine=0):
    r"""VaR estimée par simulation Monte-Carlo (rendements gaussiens)."""
    rng = np.random.default_rng(graine)
    simules = rng.normal(mu, sigma, n_sim)
    return var_historique(simules, alpha)


# ---------------------------------------------------------------------------
# 3. Cartographie des risques : criticité = probabilité × impact
# ---------------------------------------------------------------------------
def cartographie_risques(risques: pd.DataFrame) -> pd.DataFrame:
    r"""Calcule la criticité de chaque risque et son niveau.

    risques : colonnes 'risque', 'probabilite' (0-1), 'impact' (1-5).
    Criticité = probabilité × impact ; classée en Faible/Modéré/Élevé/Critique.
    """
    df = risques.copy()
    df["criticite"] = df["probabilite"] * df["impact"]

    def niveau(c):
        if c < 1.0:
            return "Faible"
        elif c < 2.0:
            return "Modéré"
        elif c < 3.5:
            return "Élevé"
        return "Critique"

    df["niveau"] = df["criticite"].apply(niveau)
    return df.sort_values("criticite", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Démonstration sur les données RRG (exercice 2023, en millions d'euros)
# ---------------------------------------------------------------------------
def _demonstration():
    print("=" * 66)
    print("ANALYSE FINANCIÈRE — Renault Retail Group (RRG), exercice 2023")
    print("=" * 66)

    bilan = {
        "actif_circulant": 1800.0, "passif_circulant": 1500.0, "stocks": 900.0,
        "dette_nette": 1200.0, "capitaux_propres": 25.4, "total_actif": 2500.0,
    }
    compte = {
        "chiffre_affaires": 3200.0, "resultat_net": -3.74,
        "ebitda": 90.0, "charges_int": 45.0,
    }
    for k, v in ratios_financiers(bilan, compte).items():
        print(f"  {k:<28}: {v:8.3f}")
    print("\n  => Gearing très élevé et capitaux propres limités : structure")
    print("     financière sous tension (forte dépendance à l'endettement).")

    print("\n" + "=" * 66)
    print("RISQUE DE MARCHÉ — VaR & Expected Shortfall (horizon journalier)")
    print("=" * 66)
    rng = np.random.default_rng(1)
    # rendements journaliers simulés (légèrement leptokurtiques)
    rendements = rng.standard_t(df=5, size=2000) * 0.012
    for a in (0.95, 0.99):
        print(f"\n  Niveau de confiance {a:.0%} :")
        print(f"    VaR historique   = {var_historique(rendements, a):.3%}")
        print(f"    VaR paramétrique = {var_parametrique(rendements, a):.3%}")
        print(f"    VaR Monte-Carlo  = {var_monte_carlo(rendements.mean(), rendements.std(), a):.3%}")
        print(f"    Expected Shortfall = {expected_shortfall(rendements, a):.3%}")

    print("\n" + "=" * 66)
    print("CARTOGRAPHIE DES RISQUES (criticité = probabilité × impact)")
    print("=" * 66)
    risques = pd.DataFrame({
        "risque": ["Solvabilité / dette", "Hausse des taux", "Chaîne d'appro.",
                   "Pénalités CO2 (CAFE)", "Cyber / RGPD", "Réputation"],
        "probabilite": [0.90, 0.75, 0.55, 0.65, 0.40, 0.30],
        "impact": [5, 4, 3, 4, 4, 3],
    })
    print(cartographie_risques(risques).to_string(index=False))


if __name__ == "__main__":
    _demonstration()
