#!/usr/bin/env bash
# ============================================================
# zenodo_batch.sh — publication en lot des 4 papiers TQNT
#   pending (P3, P4, P5, P6) sur Zenodo
#
# Prérequis :
#   1. Token Zenodo dans l'environnement :
#        export ZENODO_TOKEN="..."
#   2. Script Python zenodo_upload.py existant
#   3. Tous les .pdf + .json metadata présents
#
# Usage :
#   ./simulation/experiments/zenodo_batch.sh [--sandbox|--draft|--dry-run]
#
# Le script :
#   - Vérifie que le token est défini
#   - Vérifie l'existence de chaque fichier nécessaire
#   - Affiche un récapitulatif avant chaque upload
#   - Demande confirmation interactive avant publication réelle
#   - Affiche les DOIs obtenus à la fin
# ============================================================

set -euo pipefail

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Récupérer le mode (sandbox/draft/dry-run/prod)
MODE="${1:-prod}"
case "$MODE" in
    --sandbox|sandbox)  EXTRA_FLAG="--sandbox" ;;
    --draft|draft)      EXTRA_FLAG="--draft" ;;
    --dry-run|dry-run)  EXTRA_FLAG="--dry-run" ;;
    prod|"")            EXTRA_FLAG="" ;;
    *) echo -e "${RED}Mode inconnu : $MODE${NC}"; echo "Usage : $0 [--sandbox|--draft|--dry-run]"; exit 1 ;;
esac

# Localisation projet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJ_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  ZENODO BATCH UPLOAD — 4 papiers TQNT"
echo "════════════════════════════════════════════════════════════"
echo "  Mode      : $MODE"
echo "  Projet    : $PROJ_DIR"
echo "  Script    : simulation/experiments/zenodo_upload.py"

# Vérifier token (sauf en dry-run)
if [[ "$MODE" != "--dry-run" && "$MODE" != "dry-run" ]]; then
    if [[ -z "${ZENODO_TOKEN:-}" ]]; then
        echo -e "${RED}✗ Variable ZENODO_TOKEN non définie.${NC}"
        echo "   Obtenir un token sur https://zenodo.org/account/settings/applications/"
        echo "   Puis : export ZENODO_TOKEN=\"votre_token\""
        exit 1
    fi
    echo "  Token     : défini (${#ZENODO_TOKEN} caractères)"
fi

# Définir les 4 papiers à publier
declare -a PAPERS=(
    "paper3:dm_pearls_paper.pdf dm_pearls_paper.tex dm_pearls_paper_FR.pdf dm_pearls_paper_FR.tex"
    "paper4:topological_solar_paper_FR.pdf topological_solar_paper_FR.tex"
    "paper5:topological_battery_paper_FR.pdf topological_battery_paper_FR.tex"
    "paper6:topological_quantum_computer_paper_FR.pdf topological_quantum_computer_paper_FR.tex"
)

declare -a TITLES=(
    "P3 — Dark matter pearls (EN + FR)"
    "P4 — Conversion photovoltaïque par résonance topologique (FR)"
    "P5 — Stockage par confinement topologique (FR)"
    "P6 — Ordinateur quantique topologique nodal (FR)"
)

# Vérifier que tous les fichiers existent
echo
echo "→ Vérification des fichiers..."
ALL_OK=1
for i in "${!PAPERS[@]}"; do
    entry="${PAPERS[$i]}"
    paper_id="${entry%%:*}"
    files="${entry#*:}"

    metadata="docs/zenodo_metadata_${paper_id}.json"
    if [[ ! -f "$metadata" ]]; then
        echo -e "  ${RED}✗${NC} $metadata manquant"
        ALL_OK=0
        continue
    fi

    for f in $files; do
        if [[ ! -f "docs/$f" ]]; then
            echo -e "  ${RED}✗${NC} docs/$f manquant"
            ALL_OK=0
        fi
    done

    if [[ $ALL_OK -eq 1 ]]; then
        echo -e "  ${GREEN}✓${NC} ${TITLES[$i]}"
    fi
done

if [[ $ALL_OK -eq 0 ]]; then
    echo -e "\n${RED}✗ Certains fichiers manquent. Abandon.${NC}"
    exit 1
fi

# Confirmation interactive pour la publication réelle (pas pour sandbox/dry-run/draft)
if [[ -z "$EXTRA_FLAG" ]]; then
    echo
    echo -e "${YELLOW}⚠️  Mode PRODUCTION : publication réelle avec DOIs permanents.${NC}"
    echo "    Une fois publié, le dépôt ne peut plus être supprimé (Zenodo policy)."
    echo "    Vous pouvez seulement éditer/ajouter des métadonnées."
    echo
    read -p "Confirmer la publication des 4 papiers ? (oui/non) : " CONFIRM
    if [[ "$CONFIRM" != "oui" ]]; then
        echo "Annulé. Pour tester d'abord : $0 --sandbox"
        exit 0
    fi
fi

# Stocker les DOIs obtenus
declare -a OBTAINED_DOIS=()

# Lancer les uploads séquentiellement
echo
echo "════════════════════════════════════════════════════════════"
echo "  DÉBUT DES UPLOADS"
echo "════════════════════════════════════════════════════════════"

for i in "${!PAPERS[@]}"; do
    entry="${PAPERS[$i]}"
    paper_id="${entry%%:*}"
    files="${entry#*:}"

    echo
    echo -e "${BLUE}━━━ ${TITLES[$i]} ━━━${NC}"

    # Construire la liste --file fichier1 --file fichier2 ...
    FILE_ARGS=""
    for f in $files; do
        FILE_ARGS="$FILE_ARGS --file docs/$f"
    done

    # Exécuter le script Python et capturer la sortie
    OUTPUT_FILE="/tmp/zenodo_upload_${paper_id}.log"
    if python3 -m simulation.experiments.zenodo_upload \
        "docs/zenodo_metadata_${paper_id}.json" \
        $FILE_ARGS \
        $EXTRA_FLAG 2>&1 | tee "$OUTPUT_FILE"; then
        # Extraire le DOI obtenu (motif "DOI permanent : 10.xxxx/...")
        doi=$(grep -oE "DOI permanent : [^ ]+" "$OUTPUT_FILE" | awk '{print $4}' || true)
        if [[ -n "$doi" ]]; then
            OBTAINED_DOIS+=("$paper_id:$doi")
            echo -e "  ${GREEN}✓ DOI obtenu : $doi${NC}"
        fi
    else
        echo -e "  ${RED}✗ Échec ${paper_id}. Voir $OUTPUT_FILE pour détails.${NC}"
        echo -e "  ${YELLOW}Note : les uploads précédents restent valides.${NC}"
        exit 2
    fi
done

# Récapitulatif final
echo
echo "════════════════════════════════════════════════════════════"
echo "  RÉCAPITULATIF FINAL"
echo "════════════════════════════════════════════════════════════"
for entry in "${OBTAINED_DOIS[@]}"; do
    paper_id="${entry%%:*}"
    doi="${entry#*:}"
    echo -e "  ${GREEN}✓${NC} $paper_id : doi:$doi"
done

if [[ ${#OBTAINED_DOIS[@]} -gt 0 ]]; then
    echo
    echo "  → Étape suivante : mettre à jour les .tex avec les cross-citations"
    echo "    python3 -m simulation.experiments.update_cross_doi"
    echo
    echo "  → Ajouter les DOIs aux 'Related identifiers' des autres dépôts"
    echo "    (V5 monographe, P1, P2 via l'interface web Zenodo)"
fi

echo "════════════════════════════════════════════════════════════"
