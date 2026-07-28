"""
Theorie Nodale -- Listing des noeuds simples jusqu'a 40 croisements

Enumere tous les noeuds toriques T(p,q) avec c <= 40 et les combine
avec les 20 noeuds non-toriques du catalogue existant. Affiche un
tableau formate avec proprietes physiques (masse d'Abraham, duree de
vie P15), regles P16/P17, et sauvegarde optionnelle en CSV.

Usage :
    python3 -m simulation.experiments.knot_listing_40
"""
from __future__ import annotations

import csv
import os
import math

from theory.knot_listing import (
    find_p17_doublets,
    generate_full_listing,
)
from theory.multi_knot import sm_mass_table, sm_lifetime_table


def _fmt(val: float, width: int = 8, decimals: int = 2) -> str:
    """Formate un flottant, ou 'N/A' si NaN."""
    if math.isnan(val):
        return "N/A".rjust(width)
    return f"{val:{width}.{decimals}f}"


def _fmt_tau(log10_tau: float) -> str:
    """Formate log10(tau/s), ou 'inf' si infini."""
    if math.isinf(log10_tau):
        return "   inf"
    return f"{log10_tau:6.1f}"


def print_listing(knots: list[dict]) -> None:
    """Affiche le tableau formate sur la console."""
    header = (
        f"{'Noeud':<12} {'c':>3} {'spin':>5} {'Q':>8} "
        f"{'rho_J':>8} {'m(MeV)':>9} "
        f"{'sigma_E':>8} {'sigma_B':>8} "
        f"{'log10t':>6} {'P16':>4}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("LISTING DES NOEUDS SIMPLES — c <= 44  (masse d'Abraham, duree de vie P15)")
    print(sep)
    print(header)
    print(sep)

    for k in knots:
        p16 = "EXCL" if k["excluded"] else ""
        line = (
            f"{k['name']:<12} {k['crossing_number']:>3} "
            f"{k['spin']:>5.1f} {_fmt(k['charge'])} "
            f"{k['rho_J']:>8.3f} {k['mass_mev']:>9.4f} "
            f"{k['sigma_E']:>8.2f} {k['sigma_B']:>8.2f} "
            f"{_fmt_tau(k['log10_tau'])} {p16:>4}"
        )
        print(line)

    print(sep)
    print(f"Total : {len(knots)} noeuds")
    print(f"Masse Abraham : m_K = eta * rho_J  (eta = m_e / rho_J(3_1) = 0.1548 MeV)")
    print(f"Duree de vie  : log10(tau/s) pour n=1, k=1, canal faible P15\n")


def print_summary(knots: list[dict]) -> None:
    """Resume par famille de noeuds."""
    families: dict[str, int] = {}
    for k in knots:
        if k["is_torus"] and k["p"] is not None:
            fam = f"T({k['p']},q)"
        elif k["name"] in ("0_1",):
            fam = "unknot"
        else:
            fam = "non-torique"
        families[fam] = families.get(fam, 0) + 1

    print("RESUME PAR FAMILLE")
    print("-" * 30)
    for fam, count in sorted(families.items()):
        print(f"  {fam:<16} : {count:>3} noeuds")
    print()

    n_excl = sum(1 for k in knots if k["excluded"])
    print(f"  P16 exclus      : {n_excl}")

    # Seuil matiere noire (c* = 18)
    n_dark = sum(1 for k in knots if k["crossing_number"] > 18)
    print(f"  Candidats MN    : {n_dark}  (c > 18)")
    print()


def print_doublets(knots: list[dict]) -> None:
    """Affiche les doublets P17."""
    doublets = find_p17_doublets(knots)
    if not doublets:
        print("P17 : aucun doublet identifie.\n")
        return

    print("DOUBLETS P17 (quasi-degeneres en masse)")
    print("-" * 50)
    for a, b, gap in doublets:
        print(f"  {a:<12} <-> {b:<12}  gap = {gap:.4f} ({gap*100:.2f}%)")
    print()


def print_sm_particles() -> None:
    """Affiche la table des 13 particules SM (masse + duree de vie P15)."""
    mass_rows = sm_mass_table()
    life_rows = sm_lifetime_table()

    # Indexer les durees de vie par nom
    life_by_name = {r["name"]: r for r in life_rows}

    header = (
        f"{'Particule':<8} {'Noeuds':<35} {'k':>3} {'n':>2} "
        f"{'m_mod(MeV)':>11} {'m_SM(MeV)':>11} {'ratio':>6} "
        f"{'log10(tau_mod)':>14} {'log10(tau_exp)':>14} {'canal':>6}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("PARTICULES DU MODELE STANDARD — masse Cornell + duree de vie P15")
    print(sep)
    print(header)
    print(sep)

    for mr in mass_rows:
        name = mr["name"]
        lr = life_by_name[name]
        knots_str = "+".join(mr["knots"])
        if len(knots_str) > 35:
            knots_str = knots_str[:32] + "..."

        log_tau_mod = math.log10(lr["tau_model"]) if lr["tau_model"] > 0 else float("inf")
        if lr["tau_exp"] is not None and lr["tau_exp"] > 0:
            log_tau_exp = f"{math.log10(lr['tau_exp']):14.2f}"
        else:
            log_tau_exp = "         stable"

        if math.isinf(log_tau_mod):
            log_tau_mod_str = "           inf"
        else:
            log_tau_mod_str = f"{log_tau_mod:14.2f}"

        line = (
            f"{name:<8} {knots_str:<35} {mr['k']:>3} {len(mr['knots']):>2} "
            f"{mr['m_model']:>11.2f} {mr['m_sm']:>11.1f} {mr['ratio']:>6.3f} "
            f"{log_tau_mod_str} {log_tau_exp} {lr['channel']:>6}"
        )
        print(line)

    print(sep)
    print(f"Masse Cornell : m = eta * M_raw, eta = 0.1409 MeV (optimise log-moindres carres)")
    print(f"Duree de vie  : P15 avec canal faible (nu=7.75) ou fort/EW (nu=1.75)")
    print()


def save_csv(knots: list[dict], filepath: str) -> None:
    """Sauvegarde le listing en CSV."""
    fields = [
        "name", "crossing_number", "p", "q", "is_torus",
        "spin", "charge", "rho_J", "determinant",
        "mass_raw", "mass_mev",
        "sigma_E", "sigma_B", "int_kappa", "int_abs_tau", "L",
        "r_excl", "tau_bare", "log10_tau",
        "excluded", "has_jones",
    ]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(knots)
    print(f"CSV sauvegarde : {filepath}")


def main() -> None:
    print("Generation du listing (n_points=5000)...")
    knots = generate_full_listing(max_crossing=44, n_points=5000)

    print_listing(knots)
    print_summary(knots)
    print_doublets(knots)
    print_sm_particles()

    csv_path = os.path.join("output", "knot_listing_40.csv")
    save_csv(knots, csv_path)


if __name__ == "__main__":
    main()
