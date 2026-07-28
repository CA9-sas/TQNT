"""Schema de la trame ondulante de l'espace-temps : h, c, G + rails (23/07).

Figure de synthese fidele aux regles etablies (sec:v6-socle-ondulant + quatuor) :
  - le socle = fond d'ondes chi (niveau alpha^0), transmission aux SOMMETS
    seulement ; point bas : aucune transmission ;
  - h = la dent : graduation en ACTION (pas une maille spatiale -- aucune
    regle fixe, pas d'ether ; la LIV de reseau est exclue) ;
  - c = l'engrenement du socle lui-meme : la lumiere est un serpentin
    engrene, v = lambda f = c pour tout mode (zero dispersion) ;
  - G = l'amplitude de la trame nue (equation unique G = hbar c/M_Pl^2,
    decret 26/07) ; puits CGQ Omega^2 = 1 + kappa_g |Psi|^2, kappa_g =
    8 pi (m/M_Pl) l_P ; G nait en UNE marche a t_c ;
  - denture limite : lambda = l_P, hbar omega_socle = M_Pl (l'horloge,
    preexiste) ; masse = fraction confinee du socle.
BILINGUE (27/07) : genere la version FR (docs/socle_espace_temps.png) ET
la version EN (docs/socle_espace_temps_en.png) pour le livre anglais.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

BLEU = "#1a4a7a"
OR = "#b8860b"
VERT = "#1a6b3a"
ROUGE = "#a03030"
GRIS = "#555555"

LABELS = {
 "fr": {
  "teeth_count": "$\\sim 5\\times10^{19}$ dents",
  "gear_step": "un pas de l'engrenage $=$ $h$\n(graduation en ACTION —\naucune maille spatiale, pas d'éther)",
  "crest": "sommet : transmission (quanta, $E=hf$)",
  "trough": "point bas : aucune transmission",
  "light": "lumière : onde 3D en hélice (vis sans fin) engrenée sur les dents\n"
           r"$\Rightarrow$ $v=\lambda f=c$ pour tout mode (zéro dispersion)"
           "\n$c$ = l'engrènement de la trame de l'espace-temps elle-même — jamais de marche",
  "G_ampl": "$G$ = l'amplitude de l'onde de la\ntrame de l'espace-temps SANS particule\n"
            "($G = \\hbar c/M_{\\rm Pl}^2$ — l'équation unique ;\n"
            "$\\kappa_g(K) = 8\\pi\\,(m_K/M_{\\rm Pl})\\,\\ell_P$)",
  "compens": "effet COMPENSATOIRE :\nles voisins portent le déficit\n"
             "$\\Rightarrow$ gradient $=$ CGQ",
  "ref_dotted": "trame de l'espace-temps sans nœud (référence, pointillés)",
  "cgq": "CGQ",
  "multitude": "la particule s'assoit sur une multitude de dents —\n"
               "aucune ne tient dans un seul écartement :\n"
               "il faudrait $m \\geq M_{\\rm Pl}$ (la denture limite)",
  "reduced": "à l'emplacement du nœud : l'amplitude\n"
             "de la trame de l'espace-temps est RÉDUITE\n"
             "(le confiné = le prélevé ; exclusion P20)",
  "well": "le déficit, étalé sur $\\sim10^{19}$ dents et compensé $=$ puits CGQ : "
          "$\\Omega^2 = 1+\\kappa_g|\\Psi|^2$ ($= 1+2\\,(m/M_{\\rm Pl})\\,\\ell_P/r$ au loin — sans $G$)\n"
          "(trame de l'espace-temps déplétée $\\Leftrightarrow$ $\\Omega^2$ augmente : "
          "horloges ralenties) ; une déplétion ne peut qu'ATTIRER\n"
          "(pas d'amplitude négative à prélever) ; $G$ naît en UNE marche "
          "à $t_c$ (avant les nœuds, rien ne déplète)",
  "boxlim": "denture limite : $\\lambda = \\ell_P$\n"
            "$f_{\\rm trame} = c/\\ell_P = 1{,}855\\times10^{43}$ Hz\n"
            "$\\hbar\\,\\omega_{\\rm trame} = M_{\\rm Pl}$ (l'horloge, préexiste)\n"
            "masse d'un nœud = fraction confinée :\n"
            "$m/M_{\\rm Pl} = \\omega_0/\\omega_{\\rm trame}$",
  "quartet": "LE QUATUOR — $h$ : la dent (invariant absolu)   |   "
             "$c$ : l'engrènement (préexiste, jamais de marche)\n"
             "$M_{\\rm Pl}$ : l'horloge (préexiste, ancre)   |   "
             "$G$ : l'amplitude de la trame nue, $\\hbar c/M_{\\rm Pl}^2$ — réponse aux déficits (une marche, à $t_c$)",
  "titleA": "La trame ondulante de l'espace-temps — $h$, $c$, $G$ "
            "(engrenage gradué en action)",
  "entered": "onde ENTRÉE dans un rail : magnétisée\n"
             "(courant dirigé persistant) — se faufile",
  "latnode": "nœud du réseau : enfilement (capture magnétique,\n"
             "$\\Delta$Lk$=\\pm1$) ou casse (rencontre frontale)",
  "lock": "fermeture $\\to$ verrou :\nNŒUD $=$ particule (matière)",
  "free": "onde LIBRE de la trame de l'espace-temps :\n"
          "jamais magnétisée, jamais nouée — SPECTATRICE (la lumière)",
  "railbox": "RAIL DE LA TRAME DE L'ESPACE-TEMPS $=$ ligne nodale de\n"
             "l'ondulation stationnaire (intersection de deux plans nodaux) :\n"
             "l'ondulation y est NULLE — une onde s'y faufile sans combattre\n"
             "le milieu ; les deux pentes voisines ($+$/$-$) sont en ANTI-PHASE\n"
             "(ddp transverse, écrantée en $e^{-\\pi r/a}$ : rien à l'infini)",
  "titleB": "Les rails de la trame de l'espace-temps — vue de dessus "
            "(cellules $\\pm$ en anti-phase ; rails $=$ lignes nodales)",
  "out": "docs/socle_espace_temps.png",
  "out2": "output/socle_espace_temps.png",
 },
 "en": {
  "teeth_count": "$\\sim 5\\times10^{19}$ teeth",
  "gear_step": "one gear step $=$ $h$\n(graduation in ACTION —\nno spatial mesh, no aether)",
  "crest": "crest: transmission (quanta, $E=hf$)",
  "trough": "trough: no transmission",
  "light": "light: a 3D helical wave (endless screw) meshed on the teeth\n"
           r"$\Rightarrow$ $v=\lambda f=c$ for every mode (zero dispersion)"
           "\n$c$ = the meshing of the spacetime weave itself — never a step",
  "G_ampl": "$G$ = the amplitude of the spacetime\nweave wave WITHOUT a particle\n"
            "($G = \\hbar c/M_{\\rm Pl}^2$ — the unique equation;\n"
            "$\\kappa_g(K) = 8\\pi\\,(m_K/M_{\\rm Pl})\\,\\ell_P$)",
  "compens": "COMPENSATORY effect:\nthe neighbours carry the deficit\n"
             "$\\Rightarrow$ gradient $=$ QGF",
  "ref_dotted": "spacetime weave without a knot (reference, dotted)",
  "cgq": "QGF",
  "multitude": "the particle sits on a multitude of teeth —\n"
               "none fits within a single gap:\n"
               "it would take $m \\geq M_{\\rm Pl}$ (the limiting toothing)",
  "reduced": "at the knot's location: the amplitude\n"
             "of the spacetime weave is REDUCED\n"
             "(the confined = the withdrawn; P20 exclusion)",
  "well": "the deficit, spread over $\\sim10^{19}$ teeth and compensated $=$ QGF well: "
          "$\\Omega^2 = 1+\\kappa_g|\\Psi|^2$ ($= 1+2\\,(m/M_{\\rm Pl})\\,\\ell_P/r$ far away — without $G$)\n"
          "(depleted spacetime weave $\\Leftrightarrow$ $\\Omega^2$ increases: "
          "slowed clocks); a depletion can only ATTRACT\n"
          "(no negative amplitude to withdraw); $G$ is born in ONE step "
          "at $t_c$ (before the knots, nothing depletes)",
  "boxlim": "limiting toothing: $\\lambda = \\ell_P$\n"
            "$f_{\\rm weave} = c/\\ell_P = 1.855\\times10^{43}$ Hz\n"
            "$\\hbar\\,\\omega_{\\rm weave} = M_{\\rm Pl}$ (the clock, pre-exists)\n"
            "mass of a knot = confined fraction:\n"
            "$m/M_{\\rm Pl} = \\omega_0/\\omega_{\\rm weave}$",
  "quartet": "THE QUARTET — $h$: the tooth (absolute invariant)   |   "
             "$c$: the meshing (pre-exists, never a step)\n"
             "$M_{\\rm Pl}$: the clock (pre-exists, anchor)   |   "
             "$G$: the bare weave's amplitude, $\\hbar c/M_{\\rm Pl}^2$ — response to deficits (one step, at $t_c$)",
  "titleA": "The undulating spacetime weave — $h$, $c$, $G$ "
            "(a gear graduated in action)",
  "entered": "wave ENTERED into a rail: magnetised\n"
             "(persistent directed current) — threads its way",
  "latnode": "lattice node: threading (magnetic capture,\n"
             "$\\Delta$Lk$=\\pm1$) or break (head-on encounter)",
  "lock": "closure $\\to$ lock:\nKNOT $=$ particle (matter)",
  "free": "FREE wave of the spacetime weave:\n"
          "never magnetised, never knotted — a SPECTATOR (light)",
  "railbox": "SPACETIME-WEAVE RAIL $=$ nodal line of the\n"
             "standing undulation (intersection of two nodal planes):\n"
             "the undulation there is ZERO — a wave threads along it\n"
             "without fighting the medium; the two neighbouring slopes\n"
             "($+$/$-$) are in ANTI-PHASE (transverse p.d., screened as\n"
             "$e^{-\\pi r/a}$: nothing at infinity)",
  "titleB": "The spacetime-weave rails — top view "
            "(cells $\\pm$ in anti-phase; rails $=$ nodal lines)",
  "out": "docs/socle_espace_temps_en.png",
  "out2": "output/socle_espace_temps_en.png",
 },
}


def substrat(x):
    plateau = 1.0 / (1.0 + np.exp(-(x - 7.85) / 0.10)) \
        * 1.0 / (1.0 + np.exp((x - 10.65) / 0.10))
    ampli = (1.0
             + 0.75 * np.exp(-((x - 7.25) / 0.38) ** 2)
             + 0.75 * np.exp(-((x - 11.25) / 0.38) ** 2)
             - 0.80 * plateau)
    return 0.32 * np.sin(2 * np.pi * x) * ampli


def trefle(cx, cy, s=0.10):
    t = np.linspace(0, 2 * np.pi, 400)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    return cx + s * x, cy + s * y * 0.8


def make_figure(lang):
    L = LABELS[lang]
    fig = plt.figure(figsize=(14.5, 15.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.0], hspace=0.10)
    ax = fig.add_subplot(gs[0])
    x = np.linspace(0, 12.7, 3200)
    ys = substrat(x)

    # --- le socle (coupure d'echelle ... dans la zone depletee) ---
    ys_plot = ys.copy()
    gap = (x > 8.85) & (x < 9.65)
    ys_plot[gap] = np.nan
    ax.plot(x, ys_plot, color=BLEU, lw=2.6, zorder=3)
    ax.fill_between(x, ys - 0.05, ys + 0.05, where=~gap, color=BLEU,
                    alpha=0.10, zorder=1)
    ax.text(9.25, 0.0, "$\\cdots$", fontsize=26, color=BLEU, ha="center",
            va="center", zorder=4)
    ax.text(9.25, -0.22, L["teeth_count"], fontsize=9,
            color=BLEU, ha="center", va="top", style="italic")

    # dents aux sommets (x = 0.25 + n)
    crests = np.arange(0.25, 12.6, 1.0)
    for xc in crests:
        if 8.85 < xc < 9.65:
            continue                      # coupure d'echelle (...)
        yc = substrat(np.array([xc]))[0]
        ax.plot([xc], [yc], marker="^", ms=9, color=OR, zorder=5,
                markeredgecolor="black", markeredgewidth=0.4)

    # graduation h entre deux dents (en ACTION) -- hors zone helice
    xa, xb = 6.25, 7.25
    ya = substrat(np.array([xa]))[0] + 0.20
    ax.annotate("", xy=(xb, ya), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="<->", color=OR, lw=1.6))
    ax.text(6.75, 1.02, L["gear_step"],
            ha="center", va="bottom", fontsize=9.5, color=OR)
    ax.annotate("", xy=(6.75, ya + 0.06), xytext=(6.75, 1.00),
                arrowprops=dict(arrowstyle="->", color=OR, lw=1.1, alpha=0.7))

    # sommet / point bas
    ax.annotate(L["crest"],
                xy=(6.25, substrat(np.array([6.25]))[0] + 0.02),
                xytext=(6.05, -0.52), fontsize=10, color=VERT, ha="center",
                arrowprops=dict(arrowstyle="->", color=VERT, lw=1.2))
    ax.annotate(L["trough"],
                xy=(0.75, substrat(np.array([0.75]))[0]),
                xytext=(0.1, -0.62), fontsize=10, color=GRIS, ha="left",
                arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.2))

    # --- la lumiere : onde 3D en helice (vis sans fin) engrenee ---
    t = np.linspace(0.0, 1.0, 6000)
    x0, x1 = 0.35, 5.45
    xc = x0 + (x1 - x0) * t
    phi = 2 * np.pi * 16.0 * t          # spires serrees (vis sans fin)
    r_x, r_y = 0.135, 0.22
    xh = xc + r_x * np.sin(phi)
    yh = substrat(xc) + 0.40 + r_y * np.cos(phi)
    front = np.sin(phi) >= 0            # spires cote observateur
    for mask, lw, al, z in ((~front, 1.4, 0.50, 4), (front, 2.7, 1.0, 5)):
        xm, ym = xh.copy(), yh.copy()
        xm[~mask], ym[~mask] = np.nan, np.nan
        ax.plot(xm, ym, color=ROUGE, lw=lw, alpha=al, zorder=z,
                solid_capstyle="round")
    y_ax = substrat(xc) + 0.40
    ax.plot(xc, y_ax, color=ROUGE, lw=1.0, alpha=0.5, ls="-", zorder=3)
    ax.add_patch(FancyArrowPatch((5.45, substrat(np.array([5.45]))[0] + 0.40),
                                 (6.05, substrat(np.array([6.05]))[0] + 0.40),
                                 arrowstyle="-|>", mutation_scale=24,
                                 color=ROUGE, lw=2.0, zorder=5))
    ax.text(1.05, -1.30, L["light"], fontsize=11, color=ROUGE, ha="left")

    # --- le noeud : amplification ANNULAIRE (sommets voisins) -> G ---
    xr = np.linspace(6.9, 11.9, 700)
    yr = 0.32 * np.sin(2 * np.pi * xr)
    yr[(xr > 8.85) & (xr < 9.65)] = np.nan
    ax.plot(xr, yr, color=GRIS, lw=1.4, ls=":", zorder=2, alpha=0.9)
    y_ref = 0.32                       # amplitude du socle nu
    ax.annotate("", xy=(12.25, y_ref), xytext=(12.25, 0.0),
                arrowprops=dict(arrowstyle="<->", color=VERT, lw=1.9))
    ax.text(12.68, -0.58, L["G_ampl"],
            fontsize=9.5, color=VERT, ha="right", va="top")
    y_amp = substrat(np.array([11.25]))[0]
    ax.annotate("", xy=(11.62, y_amp * 0.97), xytext=(11.62, y_ref),
                arrowprops=dict(arrowstyle="<->", color=VERT, lw=1.6))
    ax.text(10.55, 1.02, L["compens"],
            fontsize=9.5, color=VERT, ha="center", va="bottom")
    ax.annotate(L["ref_dotted"],
                xy=(9.85, 0.32), xytext=(12.65, -2.05),
                fontsize=9.5, color=GRIS, ha="right",
                arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.0))
    for x_from, x_to in ((7.95, 8.45), (10.55, 10.05)):
        ax.add_patch(FancyArrowPatch((x_from, -0.97), (x_to, -0.97),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color=VERT, lw=1.6, zorder=4))
    ax.text(9.25, -0.95, L["cgq"], fontsize=9.5, color=VERT, ha="center",
            va="center")
    y_ctr = substrat(np.array([9.25]))[0]
    tx, ty = trefle(9.25, y_ctr + 0.72, s=0.20)
    ax.plot(tx, ty, color=VERT, lw=2.4, zorder=6)
    ax.text(9.25, -0.42, L["multitude"],
            fontsize=9.5, color=VERT, ha="center", va="top")
    ax.annotate(L["reduced"],
                xy=(8.35, substrat(np.array([8.25]))[0] + 0.05),
                xytext=(7.6, 2.42),
                fontsize=10.5, color=ROUGE, ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color=ROUGE, lw=1.4))

    # puits CGQ sous le sommet amplifie
    xw = np.linspace(7.3, 11.2, 400)
    yw = -1.15 - 0.55 * np.exp(-((xw - 9.25) / 0.85) ** 2)
    ax.plot(xw, yw, color=VERT, lw=1.8, ls="--", zorder=2)
    ax.text(6.15, -2.10, L["well"],
            fontsize=10, color=VERT, ha="center", va="top")

    # --- denture limite / M_Pl (encadre) ---
    ax.text(0.25, 1.95, L["boxlim"],
            fontsize=10.5, va="top", ha="left", color=BLEU,
            bbox=dict(boxstyle="round,pad=0.45", fc="#eef3fa", ec=BLEU, lw=1.1))

    # --- quatuor (pied de figure) ---
    ax.text(6.15, -2.95, L["quartet"],
            fontsize=10.5, ha="center", va="center", color="black",
            bbox=dict(boxstyle="round,pad=0.5", fc="#faf6ea", ec=OR, lw=1.2))

    ax.set_title(L["titleA"], fontsize=15, pad=14)
    ax.set_xlim(-0.1, 12.75)
    ax.set_ylim(-3.35, 2.5)
    ax.axis("off")

    # ================= PANNEAU B : LES RAILS DE LA TRAME =================
    ax2 = fig.add_subplot(gs[1])
    nn = 400
    Xg, Yg = np.meshgrid(np.linspace(0, 6.4, nn), np.linspace(0, 4.2, nn))
    Ug = np.cos(np.pi * Xg) * np.cos(np.pi * Yg)
    ax2.pcolormesh(Xg, Yg, Ug, cmap="RdBu", vmin=-1.9, vmax=1.9,
                   shading="auto", rasterized=True)
    for xv in np.arange(0.5, 6.4, 1.0):
        ax2.plot([xv, xv], [0, 4.2], color="black", lw=2.0, zorder=3)
    for yv in np.arange(0.5, 4.2, 1.0):
        ax2.plot([0, 6.4], [xv * 0 + yv, yv], color="black", lw=2.0, zorder=3)
    for cx in (1, 2, 3):
        for cy in (3,):
            ax2.text(cx, cy, "+" if (cx + cy) % 2 == 0 else chr(8722),
                     fontsize=15, ha="center", va="center",
                     color="black", alpha=0.65, zorder=4)
    pts = [(0.5, 0.5), (0.5, 2.5), (2.5, 2.5), (2.5, 1.5), (4.5, 1.5)]
    px = [p_[0] for p_ in pts]
    py = [p_[1] for p_ in pts]
    ax2.plot(px, py, color=VERT, lw=4.2, zorder=5, solid_capstyle="round")
    ax2.add_patch(FancyArrowPatch((4.5, 1.5), (5.05, 1.5),
                                  arrowstyle="-|>", mutation_scale=24,
                                  color=VERT, lw=3.0, zorder=6))
    ax2.annotate(L["entered"],
                 xy=(2.5, 2.0), xytext=(0.25, 3.85), fontsize=11,
                 color=VERT, ha="left",
                 arrowprops=dict(arrowstyle="->", color=VERT, lw=1.4))
    ax2.plot([2.5], [1.5], marker="o", ms=13, mfc="none", mec=OR, mew=2.6,
             zorder=6)
    ax2.annotate(L["latnode"],
                 xy=(2.5, 1.42), xytext=(0.32, 0.16), fontsize=10.5,
                 color=OR, ha="left",
                 arrowprops=dict(arrowstyle="->", color=OR, lw=1.4))
    txk, tyk = trefle(5.35, 3.05, s=0.30)
    ax2.plot(txk, tyk * 0.72 + 3.05 * 0.28, color="#0d4d2d", lw=3.2, zorder=6)
    ax2.text(5.35, 2.05, L["lock"],
             fontsize=11, color="#0d4d2d", ha="center", va="top")
    tf = np.linspace(0, 1, 600)
    xf = 0.2 + 4.5 * tf
    yf = 0.35 + 3.25 * tf
    xw2 = xf + 0.10 * np.sin(24 * np.pi * tf)
    ax2.plot(xw2, yf, color=ROUGE, lw=2.0, alpha=0.85, zorder=4)
    ax2.add_patch(FancyArrowPatch((xw2[-1], yf[-1]),
                                  (xw2[-1] + 0.3, yf[-1] + 0.16),
                                  arrowstyle="-|>", mutation_scale=18,
                                  color=ROUGE, lw=1.8, zorder=5))
    ax2.annotate(L["free"],
                 xy=(4.4, 2.63), xytext=(2.85, 3.86), fontsize=11,
                 color=ROUGE, ha="left",
                 arrowprops=dict(arrowstyle="->", color=ROUGE, lw=1.4))
    ax2.text(6.28, 0.16, L["railbox"],
             fontsize=10.5, ha="right", va="bottom", color="black",
             bbox=dict(boxstyle="round,pad=0.45", fc="#f2f2f2",
                       ec="black", lw=1.0))
    ax2.set_title(L["titleB"], fontsize=14, pad=10)
    ax2.set_xlim(-0.05, 6.45)
    ax2.set_ylim(-0.05, 4.25)
    ax2.set_aspect("equal")
    ax2.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
    for path in (L["out"], L["out2"]):
        try:
            fig.savefig(path, dpi=220)
            print("ecrit :", path)
        except Exception as e:
            print("saut :", path, e)
    plt.close(fig)


def main():
    for lang in ("fr", "en"):
        make_figure(lang)


if __name__ == "__main__":
    main()
