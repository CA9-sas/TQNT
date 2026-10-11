# -*- coding: utf-8 -*-
"""Publication du dépôt public CA9-sas/TQNT à partir du dépôt de travail.

    python3 scripts/publier_github.py            # simulation : rapport, rien n'est poussé
    python3 scripts/publier_github.py --pousser  # prépare le commit et pousse

Le dépôt public ne reçoit que ce que la liste d'INCLUSION autorise, et jamais ce que la liste
d'EXCLUSION nomme. L'exclusion l'emporte toujours. Elle protège en particulier le dossier
« processeur » (concept non protégé à ce jour : scripts p1 à p7 et brouillon associé).

Le script ne supprime rien à l'aveugle : les fichiers présents sur GitHub mais absents en local
sont listés dans le rapport, et retirés seulement avec --pousser.
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

RACINE = pathlib.Path(__file__).resolve().parents[1]
DEPOT = "git@github.com:CA9-sas/TQNT.git"
CLE = "~/.ssh/id_github_ca9"

# ---------------------------------------------------------------- ce qui part
INCLUSIONS = [
    "theory/**/*.py", "simulation/**/*.py", "tests/**/*.py", "scripts/**/*.py",
    "requirements.txt", "README.md", "LICENSE",
    # documents de référence : monographie, catalogues, publications de la collection
    "docs/TQNT_detaille_V6_new_FR.pdf", "docs/TQNT_detaille_V6_new_EN.pdf",
    "docs/TQNT_detaillee_V6_FR.pdf", "docs/TQNT_detaillee_V6_EN.pdf",
    "docs/TQNT_detaille_V7_FR.pdf", "docs/TQNT_detaille_V7_EN.pdf",
    "docs/TQNT_catalog_*.xlsx", "docs/TQNT_masses_categorisation.xlsx",
    "docs/TQNT_reassignation_V6.xlsx", "docs/TQNT_spectrum_V6.xlsx",
    "docs/catalogue_sigma_chi.csv", "docs/brouillon_catalogue_sigma_chi.tex",
    "docs/Koide_Z3_trefle.pdf", "docs/Koide_Z3_trefle_EN.pdf",
    "docs/publication_trame_espace_temps_FR.pdf", "docs/publication_trame_espace_temps_EN.pdf",
    "docs/preregistration_predictions_TQNT_2026.pdf", "docs/preregistration_predictions_TQNT_2026_EN.pdf",
    "docs/higgs_selfcoupling_paper.pdf", "docs/higgs_selfcoupling_paper_FR.pdf",
    "docs/LICENSE-CC-BY-4.0", "docs/collection_ca9wp.md", "docs/ca9wp.sty",
    "docs/knot_diagrams_v5/*.pdf", "docs/*_table.tex", "docs/entonnoir_rails*.png",
    "output/*.csv", "output/*.png", "output/*.txt", "output/*.npy",
    "docs/socle_espace_temps*.png", "docs/verification_durees_de_vie.csv",
    "docs/tqnt_logos/*", "simulation/experiments/*.sh",
    # confrontations aux données publiques (LHC, CMS, ATLAS, LHCb) : ce sont des preuves
    "docs/analyse_*.py", "docs/fit_tqnt_v3.py",
]

# ---------------------------------------------------------------- ce qui ne part JAMAIS
EXCLUSIONS = [
    # ---- concept de processeur : confidentiel tant qu'il n'est pas protégé
    r"simulation/experiments/p[1-9]_.*\.py$",
    r".*processeur_triangle_trefles.*",
    r".*brouillon_qubit_enlacement.*",
    # ---- secrets, données lourdes, documents privés
    r"\.env", r".*/\.env.*", r"docs/references/AGBT19B_334/.*", r".*\.(fits|tar|gz|zip)$",
    r"docs/.*submission/.*", r"docs/soumissions/.*", r"docs/Archives_recherches/.*",
    r"docs/plans/.*", r"docs/3d_prints/.*", r"docs/brevet.*", r".*lettre.*\.pdf$",
    # applications et dossiers de brevet : hors du périmètre scientifique public
    r"docs/generate_.*\.py$", r"docs/gen_patent_figs\.py$",
    # monographie V7 : référence de travail, non diffusée à ce jour
    r"docs/TQNT_detaille_V7_.*",
    r".*\.aux$", r".*\.log$", r".*\.synctex\.gz$", r"__pycache__", r"\.pytest_cache",
]


def exclu(rel):
    return any(re.search(m, rel) for m in EXCLUSIONS)


def fichiers_a_publier():
    retenus = set()
    for motif in INCLUSIONS:
        for p in RACINE.glob(motif):
            if not p.is_file():
                continue
            rel = str(p.relative_to(RACINE))
            if not exclu(rel):
                retenus.add(rel)
    return sorted(retenus)


def etat_distant(tmp):
    subprocess.run(["git", "clone", "--depth", "1", DEPOT, str(tmp)], check=True,
                   env={"GIT_SSH_COMMAND": f"ssh -i {pathlib.Path(CLE).expanduser()} -o IdentitiesOnly=yes",
                        "PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home())},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {str(p.relative_to(tmp)) for p in tmp.rglob("*")
            if p.is_file() and ".git/" not in str(p.relative_to(tmp))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pousser", action="store_true", help="écrit et pousse le commit")
    ap.add_argument("--retirer", action="store_true",
                    help="supprime aussi, sur le dépôt public, les fichiers absents en local "
                         "(par défaut ils sont conservés : README.md et LICENSE n'existent que là-bas)")
    ap.add_argument("--message", default=None)
    a = ap.parse_args()

    locaux = fichiers_a_publier()
    bloques = [str(p.relative_to(RACINE)) for p in RACINE.rglob("*")
               if p.is_file() and exclu(str(p.relative_to(RACINE)))
               and re.search(r"p[1-9]_|processeur_triangle|qubit_enlacement", str(p))]

    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d) / "depot"
        try:
            distants = etat_distant(tmp)
        except subprocess.CalledProcessError:
            print("Clonage impossible : vérifier la clé de déploiement", CLE)
            return 1

        nouveaux = [f for f in locaux if f not in distants]
        communs = [f for f in locaux if f in distants]
        modifies = [f for f in communs
                    if (RACINE / f).read_bytes() != (tmp / f).read_bytes()]
        retires = sorted(f for f in distants if f not in set(locaux))

        print("=" * 76)
        print(f"À publier : {len(locaux)} fichiers | nouveaux {len(nouveaux)} | "
              f"modifiés {len(modifies)} | absents en local {len(retires)}")
        print("=" * 76)
        for titre, liste in (("NOUVEAUX", nouveaux), ("MODIFIÉS", modifies),
                             ("ABSENTS EN LOCAL (conservés sauf --retirer)", retires)):
            print(f"\n{titre} ({len(liste)}) :")
            for f in liste[:25]:
                print("   ", f)
            if len(liste) > 25:
                print(f"    … et {len(liste) - 25} autres")
        print(f"\nBLOQUÉS PAR L'EXCLUSION (confidentiels) : {len(bloques)}")
        for f in bloques:
            print("   ", f)

        if not a.pousser:
            print("\nSimulation : rien n'a été poussé. Relancer avec --pousser.")
            return 0

        if a.retirer:
            for f in retires:
                (tmp / f).unlink()
        elif retires:
            print(f"({len(retires)} fichiers distants conservés — relancer avec --retirer pour les supprimer)")
        for f in locaux:
            cible = tmp / f
            cible.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RACINE / f, cible)
        env = {"GIT_SSH_COMMAND": f"ssh -i {pathlib.Path(CLE).expanduser()} -o IdentitiesOnly=yes",
               "PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home())}
        msg = a.message or "TQNT — mise à jour du code, des tests et des documents de référence"
        subprocess.run(["git", "add", "-A"], cwd=tmp, check=True)
        # le .gitignore recopié depuis le dépôt de travail ignore output/ : la liste
        # d'inclusion fait autorité, on force l'ajout des fichiers retenus.
        for lot in [locaux[i:i + 200] for i in range(0, len(locaux), 200)]:
            subprocess.run(["git", "add", "-f", "--"] + lot, cwd=tmp, check=True)
        subprocess.run(["git", "-c", "user.name=CA9 SAS", "-c", "user.email=lilian.cariou@ca9.fr",
                        "commit", "-m", msg], cwd=tmp, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=tmp, check=True, env=env)
        print("\nPoussé sur CA9-sas/TQNT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
