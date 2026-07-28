"""
ame2020_load.py — Parseur AME 2020 (Atomic Mass Evaluation) pour TQNT-V5.

Contexte TQNT-V5
----------------
Dans la TQNT-V5 (cf. docs/TQNT_detaillee_V5_FR.tex), la masse d'un objet
topologique (nœud ou entrelacs) est donnée canoniquement par l'opérateur

    m(L) = Ω_ts(L)                              (forme V5, chap. 14)

et asymptotiquement par la forme historique V4 :

    m(L) = α₀ · U_EM(L) · c_J(L)^{3,55}         (~5 % de précision)

avec α₀ = 8,16 × 10⁻⁴ MeV calibré sur l'électron.

Le nucléon V5 est un entrelacs 3₁ + 7₂ + 7₂ ; m_p = 938,54 MeV, m_n = 939,83 MeV.
Pour un NOYAU nucléaire (¹H, ⁴He, ¹²C, …) chaque nucléon est un sous-entrelacs ;
l'énergie de liaison nucléaire doit émerger collectivement de la métrique
émergente (P5) et de la tension d'enlacement (β harmonique). Cette extension
n'est PAS encore implémentée — ce module fournit l'INFRASTRUCTURE DE DONNÉES
nécessaire pour confronter une future prédiction TQNT-V5 à l'AME 2020.

Source des données
------------------
Huang, Wang, Kondev, Audi, Naimi, "The Ame2020 atomic mass evaluation (I)",
Chinese Physics C45, 030002 (2021).
Fichier : /mnt/data/data_recherche/ame2020/mass_1.mas20.txt
Format Fortran (lignes de données) :
    a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,1x,f10.5,1x,a2,f13.5,f11.5,1x,i3,1x,f13.6,f12.6
soit
    cc N-Z  N  Z  A  EL  O  MASS_EXCESS  DMASS  BE/A  DBE/A  B  E_beta  DE_beta  A_atomic_int A_atomic_micro_u  dA

Marqueurs AME :
    '#' à la place de la décimale : valeur ESTIMÉE (extrapolation)
    '*' à la place de la valeur   : quantité NON CALCULABLE
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_AME_PATH = Path("/mnt/data/data_recherche/ame2020/mass_1.mas20.txt")
DEFAULT_NUBASE_PATH = Path("/mnt/data/data_recherche/ame2020/nubase_4.mas20.txt")

# Colonnes fixes (indices 0-based, fins exclusives) déduites du format Fortran AME 2020.
# Vérifiées par inspection ligne à ligne du fichier mass_1.mas20.txt (cf. en-tête du fichier).
_COL_NMZ = (1, 4)            # N-Z         (i3)
_COL_N = (4, 9)              # N           (i5)
_COL_Z = (9, 14)             # Z           (i5)
_COL_A = (14, 19)            # A           (i5)
_COL_EL = (20, 23)           # element     (a3)
_COL_ORIGIN = (23, 27)       # origin tag  (a4)
_COL_ME = (28, 42)           # mass excess (f14.6, keV)
_COL_DME = (42, 54)          # mass excess unc (f12.6, keV)
_COL_BE_A = (54, 67)         # binding energy / A (f13.5, keV)
_COL_DBE_A = (68, 78)        # binding energy / A unc (f10.5, keV)
_COL_B_TYPE = (79, 81)       # beta-decay type (a2)
_COL_EBETA = (81, 94)        # beta-decay energy (f13.5, keV)
_COL_DEBETA = (94, 105)      # beta-decay energy unc (f11.5, keV)
_COL_AM_INT = (106, 109)     # atomic mass integer part (i3, u)
_COL_AM_FRAC = (110, 123)    # atomic mass fractional (f13.6, micro-u)
_COL_DAM = (123, 135)        # atomic mass uncertainty (f12.6, micro-u)


def _parse_ame_float(raw: str) -> tuple[float, bool]:
    """Convertit une cellule AME en float.

    Renvoie (valeur, is_extrapolated).
    Retourne (np.nan, False) si la cellule est vide ou marquée '*' (non calculable).
    Si la cellule contient '#' (extrapolation systématique), is_extrapolated=True
    et le '#' est remplacé par '.' avant conversion.
    """
    s = raw.strip()
    if s == "" or "*" in s:
        return (np.nan, False)
    extrap = "#" in s
    if extrap:
        s = s.replace("#", ".")
    # Certaines cellules d'incertitude AME sont des entiers suivis de '#'
    # (ex : "2000#") — après remplacement on a "2000." qui parse correctement.
    try:
        return (float(s), extrap)
    except ValueError:
        return (np.nan, extrap)


def _parse_int(raw: str) -> int | None:
    s = raw.strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _is_data_line(line: str) -> bool:
    """Une ligne de données AME a une colonne A (cols 14:19) numérique."""
    if len(line) < 20:
        return False
    try:
        int(line[_COL_A[0] : _COL_A[1]].strip())
        int(line[_COL_Z[0] : _COL_Z[1]].strip())
        int(line[_COL_N[0] : _COL_N[1]].strip())
    except ValueError:
        return False
    return True


def _load_halflives(path: Path = DEFAULT_NUBASE_PATH) -> dict[tuple[int, int], str]:
    """Charge le ground-state half-life depuis NUBASE indexé par (A, Z).

    Renvoie un dict { (A,Z) : "valeur unité" } (string brute) ; clés
    uniquement pour les états fondamentaux (i=0 dans le code ZZZi).
    Si NUBASE est absent, renvoie un dict vide.
    """
    out: dict[tuple[int, int], str] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("#") or len(line) < 80:
                continue
            try:
                A = int(line[0:3].strip())
                zzzi = line[4:8].strip()
                Z = int(zzzi[:-1])
                i = int(zzzi[-1])
            except ValueError:
                continue
            if i != 0:
                continue  # on ignore isomères/niveaux
            t_val = line[69:78].strip()
            t_unit = line[78:80].strip()
            if t_val == "":
                continue
            out[(A, Z)] = (t_val + " " + t_unit).strip()
    return out


def load_ame2020(
    path: Path | str = DEFAULT_AME_PATH,
    keep_extrapolated: bool = False,
    attach_halflife: bool = True,
    nubase_path: Path | str = DEFAULT_NUBASE_PATH,
) -> pd.DataFrame:
    """Parse le fichier AME 2020 mass_1.mas20.txt en DataFrame.

    Paramètres
    ----------
    path : chemin du fichier mass_1.mas20.txt
    keep_extrapolated : si False, filtre les noyaux dont mass_excess ou BE/A
        sont marqués '#' (estimations non expérimentales). Par défaut False.
    attach_halflife : si True, joint la demi-vie ground-state depuis NUBASE.
    nubase_path : chemin du fichier NUBASE.

    Colonnes du DataFrame
    ---------------------
    N, Z, A                            : entiers
    element                            : symbole chimique (str)
    mass_excess_keV, mass_excess_unc_keV
    binding_E_per_A_keV, binding_E_per_A_unc_keV
    binding_E_total_keV                : BE/A × A (énergie totale de liaison)
    atomic_mass_uu                     : masse atomique en micro-u (1e-6 u)
    atomic_mass_unc_uu
    extrapolated                       : True si mass_excess ou BE/A AME-extrapolé
    half_life                          : str brute NUBASE ("stbl", "12.32 y", …)
                                         (NaN si non joint)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AME 2020 file not found: {path}")

    rows: list[dict] = []
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not _is_data_line(line):
                continue
            # padding pour éviter IndexError sur lignes courtes
            line = line.ljust(140)

            N = _parse_int(line[_COL_N[0] : _COL_N[1]])
            Z = _parse_int(line[_COL_Z[0] : _COL_Z[1]])
            A = _parse_int(line[_COL_A[0] : _COL_A[1]])
            element = line[_COL_EL[0] : _COL_EL[1]].strip()
            me, me_ext = _parse_ame_float(line[_COL_ME[0] : _COL_ME[1]])
            dme, _ = _parse_ame_float(line[_COL_DME[0] : _COL_DME[1]])
            bea, bea_ext = _parse_ame_float(line[_COL_BE_A[0] : _COL_BE_A[1]])
            dbea, _ = _parse_ame_float(line[_COL_DBE_A[0] : _COL_DBE_A[1]])
            am_int = _parse_int(line[_COL_AM_INT[0] : _COL_AM_INT[1]])
            am_frac, _ = _parse_ame_float(line[_COL_AM_FRAC[0] : _COL_AM_FRAC[1]])
            dam, _ = _parse_ame_float(line[_COL_DAM[0] : _COL_DAM[1]])

            if am_int is not None and not np.isnan(am_frac):
                atomic_mass_uu = am_int * 1_000_000.0 + am_frac
            else:
                atomic_mass_uu = np.nan
            be_total = bea * A if (A is not None and not np.isnan(bea)) else np.nan

            rows.append(
                {
                    "N": N,
                    "Z": Z,
                    "A": A,
                    "element": element,
                    "mass_excess_keV": me,
                    "mass_excess_unc_keV": dme,
                    "binding_E_per_A_keV": bea,
                    "binding_E_per_A_unc_keV": dbea,
                    "binding_E_total_keV": be_total,
                    "atomic_mass_uu": atomic_mass_uu,
                    "atomic_mass_unc_uu": dam,
                    "extrapolated": bool(me_ext or bea_ext),
                }
            )

    df = pd.DataFrame(rows)

    if attach_halflife:
        hl = _load_halflives(Path(nubase_path))
        if hl:
            df["half_life"] = [
                hl.get((int(a), int(z)), np.nan) if pd.notna(a) and pd.notna(z) else np.nan
                for a, z in zip(df["A"], df["Z"])
            ]
        else:
            df["half_life"] = np.nan
    else:
        df["half_life"] = np.nan

    if not keep_extrapolated:
        df = df.loc[~df["extrapolated"]].reset_index(drop=True)

    return df


def predict_mass_V5(Z: int, N: int) -> float:
    """Stub TQNT-V5 — opérateur de masse axiomatique Ω_ts non encore implémenté.

    La forme canonique V5 est m(L) = Ω_ts(L) (chap. 14 de
    docs/TQNT_detaillee_V5_FR.tex). Pour un noyau (A=N+Z) il faut décomposer
    en sous-entrelacs nucléoniques (3₁+7₂+7₂ pour chaque proton et neutron,
    cf. CLAUDE.md) puis appliquer l'opérateur Ω_ts collectif sur l'entrelacs
    composite — ce qui n'a pas encore d'implémentation numérique.

    Renvoie np.nan + warning.
    """
    warnings.warn(
        "predict_mass_V5: not yet implemented — see TQNT_detaillee_V5_FR.tex "
        "ch. 14 for axiomatic mass operator Ω_ts. Returning NaN.",
        stacklevel=2,
    )
    return np.nan


def binding_energy_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Courbe semi-empirique BE/A (keV) en fonction de A.

    Pour chaque A on prend l'isotope LE PLUS LIÉ (max BE/A), ce qui reproduit
    la courbe canonique de Bethe–Weizsäcker (pic à ⁵⁶Fe ~ 8794 keV/nucléon).
    """
    sub = df.dropna(subset=["A", "binding_E_per_A_keV"])
    sub = sub.loc[sub["binding_E_per_A_keV"] > 0]
    grouped = sub.groupby("A")["binding_E_per_A_keV"].max().sort_index()
    return grouped.index.to_numpy(dtype=int), grouped.to_numpy(dtype=float)


def _stats(df: pd.DataFrame) -> dict:
    return {
        "n_nuclei": len(df),
        "Z_min": int(df["Z"].min()),
        "Z_max": int(df["Z"].max()),
        "A_min": int(df["A"].min()),
        "A_max": int(df["A"].max()),
        "N_max": int(df["N"].max()),
        "n_extrapolated_in_full": None,
        "max_BE_per_A_keV": float(df["binding_E_per_A_keV"].max()),
    }


def _main() -> None:
    df_full = load_ame2020(keep_extrapolated=True)
    df = load_ame2020(keep_extrapolated=False)

    stats = _stats(df)
    stats["n_extrapolated_in_full"] = int(df_full["extrapolated"].sum())

    print("=" * 64)
    print("AME 2020 — chargement réussi")
    print("=" * 64)
    print(f"Source : {DEFAULT_AME_PATH}")
    print(f"Noyaux total (avec extrapolés) : {len(df_full)}")
    print(f"Noyaux expérimentaux (filtrés) : {stats['n_nuclei']}")
    print(f"Noyaux extrapolés ('#') exclus : {stats['n_extrapolated_in_full']}")
    print(f"Plage Z : {stats['Z_min']} → {stats['Z_max']}")
    print(f"Plage A : {stats['A_min']} → {stats['A_max']}")
    print(f"N max   : {stats['N_max']}")
    print(f"BE/A max (keV/nucléon) : {stats['max_BE_per_A_keV']:.3f}")
    n_hl = int(df["half_life"].notna().sum())
    print(f"Demi-vies jointes depuis NUBASE : {n_hl}/{len(df)}")

    A_arr, BE_arr = binding_energy_curve(df)
    print(f"Courbe BE/A vs A : {len(A_arr)} points (A = {A_arr.min()}…{A_arr.max()})")

    # Tracé
    out_dir = Path("/home/euk9/projets/physique_nouvelle/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ame2020_binding_energy.png"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(A_arr, BE_arr / 1000.0, lw=1.0, color="#1f4e79", label="AME 2020 (max BE/A par A)")
    i_peak = int(np.argmax(BE_arr))
    ax.axhline(BE_arr[i_peak] / 1000.0, color="#c0392b", lw=0.7, ls="--", alpha=0.6,
               label=f"max BE/A = {BE_arr[i_peak]/1000:.3f} MeV/A (A={A_arr[i_peak]})")
    ax.scatter([A_arr[i_peak]], [BE_arr[i_peak] / 1000.0], color="#c0392b", zorder=5,
               s=30)
    ax.set_xlabel("Nombre de masse A")
    ax.set_ylabel("Énergie de liaison par nucléon BE/A (MeV)")
    ax.set_title("Courbe d'énergie de liaison nucléaire — AME 2020\n"
                 "(infrastructure pour prédiction TQNT-V5 future)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, A_arr.max() + 5)
    ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Graphique : {out_path}")


if __name__ == "__main__":
    _main()
