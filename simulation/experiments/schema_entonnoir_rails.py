"""Schema de l'entonnoir : du faufilement aux planchers (23/07).

Figure pour V6_new : la trame de l'espace-temps ondule -> les ondes
entrent dans les rails (magnetisation) -> faufilement/fermetures ->
spectre brut (trefle dominant, queue sombre) -> entonnoir (cascade
P15, potentiel d'enlacement, conservation de Q) -> les planchers
(electron, proton) + le reservoir sombre + la lumiere a part.
Sortie : docs/entonnoir_rails.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLEU = "#1a4a7a"
OR = "#b8860b"
VERT = "#1a6b3a"
ROUGE = "#a03030"
VIOLET = "#5b3a8a"
GRIS = "#555555"


def boite(ax, x, y, w, h, txt, fc, ec, fs=10.5, tc="black", lw=1.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4)


def fleche(ax, x0, y0, x1, y1, c, lw=2.2, txt=None, dx=0.14, fs=9.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=20, color=c, lw=lw, zorder=2))
    if txt:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, txt, fontsize=fs,
                color=c, ha="left", va="center", zorder=4)


LABELS = {
    "fr": {
        "trame": "LA TRAME DE L'ESPACE-TEMPS ONDULE\n"
                 "(rails $=$ lignes nodales ; graduation en action)",
        "entree_fl": "entrée dans un rail :\nMAGNÉTISATION",
        "entree": "onde ENTRÉE : courant persistant\nse faufile (fantôme : superposition)",
        "libre": "onde LIBRE : spectatrice\n$=$ LA LUMIÈRE ($\\mathcal{S}=0$)",
        "libre_note": "jamais magnétisée, jamais nouée,\njamais massive — traverse tout",
        "verrou": "fermeture $\\to$ VERROU (nœud)\n$P = 1-e^{-(N-22)/126}$, seuil $\\sim$24 pas",
        "spectre": "LE SPECTRE BRUT DES NŒUDS\n"
                   "trèfle $3_1$ : $59\\,\\%$ (le premier produit)  |  $4_1$, $5_2$, "
                   "$6_x$... : transitoires  |  queue $c\\geq23$ : $\\sim10^{-7}$",
        "entonnoir": "L'ENTONNOIR\n"
                     "cascade P15 (les transitoires se défont) ; potentiel d'enlacement "
                     "$\\beta(\\mathrm{Lk}-3)^2$\n(marche $=1{,}29$ MeV $=\\Delta m(n-p)$) ; "
                     "conservation de $Q$ ($n_e = n_p$)",
        "survie": "nœuds isolés $c\\geq23$ :\nsurvie (stabilité CS)",
        "electron": "ÉLECTRON\nplancher leptonique\n($3_1$, modes $Z_3$ $\\to e$)",
        "proton": "PROTON (/neutron lié)\nplancher baryonique\n"
                  "($3_1{+}7_2{+}\\bar7_2$, Lk$=3$)",
        "dm": "MATIÈRE NOIRE\nréservoir $T(2,q{\\geq}23)$\n(neutre, gravité seule)",
        "pied": "la combinatoire des rails fixe QUI PEUT naître "
                "(rapport brut $e/p \\sim 3\\times10^{7}$) ;\n"
                "l'entonnoir et la conservation fixent QUI SURVIT : "
                "la hiérarchie observée ($e$, $p$ stables — tout le reste transitoire)",
        "titre": "L'entonnoir : du faufilement dans les rails de la trame "
                 "de l'espace-temps aux planchers de la matière",
        "out": "entonnoir_rails.png",
    },
    "en": {
        "trame": "THE SPACETIME WEAVE UNDULATES\n"
                 "(rails $=$ nodal lines ; graduation in action)",
        "entree_fl": "entry into a rail:\nMAGNETIZATION",
        "entree": "ENTERED wave: persistent current\nthreads through (ghost: superposition)",
        "libre": "FREE wave: spectator\n$=$ LIGHT ($\\mathcal{S}=0$)",
        "libre_note": "never magnetized, never knotted,\nnever massive — passes through everything",
        "verrou": "closure $\\to$ LOCK (knot)\n$P = 1-e^{-(N-22)/126}$, threshold $\\sim$24 steps",
        "spectre": "THE RAW KNOT SPECTRUM\n"
                   "trefoil $3_1$: $59\\,\\%$ (the first product)  |  $4_1$, $5_2$, "
                   "$6_x$...: transients  |  tail $c\\geq23$: $\\sim10^{-7}$",
        "entonnoir": "THE FUNNEL\n"
                     "P15 cascade (transients come undone) ; linking potential "
                     "$\\beta(\\mathrm{Lk}-3)^2$\n(step $=1.29$ MeV $=\\Delta m(n-p)$) ; "
                     "conservation of $Q$ ($n_e = n_p$)",
        "survie": "isolated knots $c\\geq23$:\nsurvival (CS stability)",
        "electron": "ELECTRON\nleptonic floor\n($3_1$, $Z_3$ modes $\\to e$)",
        "proton": "PROTON (/bound neutron)\nbaryonic floor\n"
                  "($3_1{+}7_2{+}\\bar7_2$, Lk$=3$)",
        "dm": "DARK MATTER\nreservoir $T(2,q{\\geq}23)$\n(neutral, gravity only)",
        "pied": "the rail combinatorics fixes WHO CAN be born "
                "(raw ratio $e/p \\sim 3\\times10^{7}$) ;\n"
                "the funnel and conservation fix WHO SURVIVES: "
                "the observed hierarchy ($e$, $p$ stable — everything else transient)",
        "titre": "The funnel: from threading in the rails of the spacetime "
                 "weave to the floors of matter",
        "out": "entonnoir_rails_en.png",
    },
}


def draw(L):
    fig, ax = plt.subplots(figsize=(12.8, 9.6))

    # niveau 0 : la trame
    boite(ax, 5.0, 9.30, 8.6, 0.85, L["trame"], "#eef3fa", BLEU, fs=12, lw=1.8)

    # bifurcation : entree / spectatrice
    fleche(ax, 3.4, 8.82, 2.4, 8.05, VERT, 2.6, L["entree_fl"], dx=-2.75)
    fleche(ax, 6.6, 8.82, 7.9, 8.05, ROUGE, 2.2)
    boite(ax, 2.4, 7.60, 4.3, 0.80, L["entree"], "#eaf4ec", VERT, fs=10)
    boite(ax, 8.15, 7.60, 3.4, 0.80, L["libre"], "#f8ecec", ROUGE, fs=10)
    ax.text(8.15, 6.98, L["libre_note"], fontsize=8.8, color=ROUGE,
            ha="center", va="top", style="italic")

    # niveau 2 : fermetures -> spectre brut
    fleche(ax, 2.4, 7.18, 2.4, 6.45, VERT, 2.6, L["verrou"], dx=0.22)
    boite(ax, 4.05, 5.85, 7.7, 1.00, L["spectre"], "#faf6ea", OR, fs=10.5)

    # niveau 3 : l'entonnoir
    for x0 in (1.6, 4.05, 6.5):
        fleche(ax, x0, 5.32, 4.05 + (x0 - 4.05) * 0.42, 4.62, OR, 2.0)
    boite(ax, 4.05, 4.10, 7.4, 1.02, L["entonnoir"], "#f4f0fa", VIOLET, fs=10.5)

    # niveau 4 : les planchers + le reservoir
    fleche(ax, 2.6, 3.56, 1.6, 2.82, VERT, 2.6)
    fleche(ax, 4.35, 3.56, 4.55, 2.80, VERT, 2.6)
    fleche(ax, 6.7, 5.30, 8.30, 2.82, VIOLET, 2.0)
    ax.text(9.30, 4.00, L["survie"], fontsize=9.5, color=VIOLET,
            ha="center", va="center", zorder=4)
    boite(ax, 1.50, 2.30, 2.55, 0.92, L["electron"], "#eaf4ec", VERT, fs=10.5, lw=1.8)
    boite(ax, 4.62, 2.30, 2.85, 0.92, L["proton"], "#eaf4ec", VERT, fs=10.5, lw=1.8)
    boite(ax, 8.30, 2.30, 3.05, 0.92, L["dm"], "#f4f0fa", VIOLET, fs=10.5, lw=1.8)

    # pied
    ax.text(5.0, 1.10, L["pied"], fontsize=11, ha="center", va="center",
            color="black",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f5f5f5", ec=GRIS, lw=1.1))

    ax.set_title(L["titre"], fontsize=14, pad=12)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0.55, 9.85)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
    for base in ("docs/", "output/"):
        path = base + L["out"]
        try:
            fig.savefig(path, dpi=200)
            print("ecrit :", path)
        except Exception as e:
            print("saut :", path, e)
    plt.close(fig)


def main():
    for lang in ("fr", "en"):
        draw(LABELS[lang])


if __name__ == "__main__":
    main()
