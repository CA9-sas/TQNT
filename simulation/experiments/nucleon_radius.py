"""
Expérience : Rayon des nœuds du nucléon par densité particulaire

Calcule les rayons libres et comprimés des 3 nœuds du nucléon
(9₁ + 7₄ + 6₃), la densité particulaire, la correction métrique,
et la prédiction du rayon du proton.

Génère :
  - output/rp_vs_density.pdf (courbe r_p vs densité avec limites)
  - docs/brouillon.pdf (document LaTeX complet)
"""
import subprocess
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.nucleon_radius import (
    full_nucleon_analysis,
    proton_radius_vs_density,
    NUCLEON_KNOTS, NUCLEON_K, ACTIVE_KNOTS, N_PSI,
    HBAR_C_MEV_FM, M_P_MEV, R_P_EXP_FM,
)

# Répertoires de sortie
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DOCS = os.path.join(_ROOT, "docs")
_OUTPUT = os.path.join(_ROOT, "output")


def generate_figure(curve: dict, fig_path: str):
    """Génère la figure r_p vs densité particulaire avec limites."""
    n = curve["n"]
    r_p = curve["r_p_fm"]
    n_star = curve["n_star"]
    r_p_star = curve["r_p_star"]
    r_schwarzschild = curve["r_schwarzschild"]
    r_compton = curve["r_compton"]
    r_p_exp = curve["r_p_exp"]
    P = curve["P"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panneau gauche : r_p vs densité ──
    ax1.loglog(n, r_p, "b-", linewidth=2, label=r"$r_p(n) = 4\hbar c\,/\,m(d)\,c^2$")

    # Point d'équilibre (nucléon physique)
    ax1.plot(n_star, r_p_star, "ro", markersize=10, zorder=5,
             label=f"Nucléon ($r_p = {r_p_star:.3f}$ fm)")

    # Limites horizontales
    ax1.axhline(r_p_exp, color="green", ls="--", lw=1.5, alpha=0.8,
                label=f"$r_p$ exp. = {r_p_exp} fm")
    ax1.axhline(r_compton, color="orange", ls=":", lw=1.5, alpha=0.8,
                label=rf"$\lambda_C = \hbar/(m_p c)$ = {r_compton:.4f} fm")
    ax1.axhline(r_schwarzschild, color="red", ls="-.", lw=1.5, alpha=0.8,
                label=f"$r_S$ = {r_schwarzschild:.2e} fm")

    # Zones
    ax1.axhspan(r_compton, r_p.max() * 2, alpha=0.05, color="blue",
                label="Zone classique")
    ax1.axhspan(r_schwarzschild, r_compton, alpha=0.05, color="orange",
                label="Zone quantique")

    ax1.set_xlabel(r"Densité particulaire $n$ (unités code$^{-3}$)", fontsize=12)
    ax1.set_ylabel(r"Rayon du proton $r_p$ (fm)", fontsize=12)
    ax1.set_title("Rayon du proton vs. densité particulaire", fontsize=13)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(r_p.min() * 0.3, r_p.max() * 3)
    ax1.grid(True, alpha=0.3, which="both")

    # Annotations
    ax1.annotate("Équilibre\nnucléon",
                 xy=(n_star, r_p_star),
                 xytext=(n_star * 0.02, r_p_star * 2),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=9, color="red", ha="center")

    # ── Panneau droit : r_p vs pression ──
    # On ne garde que les points à d < d* (pression positive = compression)
    d_arr = curve["d"]
    d_star = curve["d_star"]
    mask_comp = d_arr < d_star * 0.99  # compression
    mask_tens = d_arr > d_star * 1.01  # tension

    if np.any(mask_comp):
        ax2.loglog(P[mask_comp], r_p[mask_comp], "b-", linewidth=2,
                   label="Compression ($d < d^*$)")
    if np.any(mask_tens):
        ax2.loglog(-P[mask_tens], r_p[mask_tens], "r--", linewidth=2,
                   label="Tension ($d > d^*$)")

    # Point d'équilibre (P=0, on place à P_min visible)
    P_eq_display = np.min(np.abs(P[P != 0])) if np.any(P != 0) else 1.0
    ax2.plot(P_eq_display, r_p_star, "ro", markersize=10, zorder=5,
             label=f"Équilibre ($P=0$, $r_p = {r_p_star:.3f}$ fm)")

    # Limites
    ax2.axhline(r_p_exp, color="green", ls="--", lw=1.5, alpha=0.8)
    ax2.axhline(r_compton, color="orange", ls=":", lw=1.5, alpha=0.8)

    ax2.set_xlabel(r"Pression $|P|$ (unités code$^{-4}$)", fontsize=12)
    ax2.set_ylabel(r"Rayon du proton $r_p$ (fm)", fontsize=12)
    ax2.set_title("Rayon du proton vs. pression particulaire", fontsize=13)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def generate_latex(result: dict, curve: dict, fig_relpath: str) -> str:
    """Génère le source LaTeX du brouillon."""
    eq = result["cornell"]
    radii = result["knot_radii"]
    rp = result["proton_radius"]
    L0 = result["L0_fm"]

    radii_fm = []
    for kr in radii:
        radii_fm.append({
            **kr,
            "r_free_fm": kr["r_free"] * L0,
            "r_conf_fm": kr["r_conf"] * L0,
            "r_proper_fm": kr["r_proper"] * L0,
        })

    tex = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[margin=2.5cm]{geometry}
\usepackage{hyperref}
\usepackage[most]{tcolorbox}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{float}
\newcommand{\dd}{\mathrm{d}}

\sisetup{
  output-decimal-marker={,},
  per-mode=symbol,
  exponent-product=\cdot,
}

\title{Rayon des n\oe uds du nucl\'eon\\par densit\'e particulaire (compression)}
\author{Th\'eorie Quantique Nodale}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
On calcule le rayon effectif de chaque n\oe ud ($9_1$, $7_4$, $6_3$) du
nucl\'eon, libre et comprim\'e par le potentiel de Cornell topologique (P12).
La densit\'e particulaire au sein de chaque n\oe ud est d\'eriv\'ee, puis
corrig\'ee par le facteur conforme de la m\'etrique \'emergente (P5).
La pr\'ediction du rayon de charge du proton donne
$r_p = """ + f"{rp['r_p_fm']:.4f}" + r"""~\text{fm}$,
en accord avec la valeur exp\'erimentale
$""" + f"{R_P_EXP_FM}" + r"""~\text{fm}$ (\`a $""" + f"{rp['error_pct']:.2f}" + r"""\,\%$ pr\`es).
On trace ensuite la courbe $r_p(n)$ en fonction de la densit\'e
particulaire, avec les limites de Schwarzschild et de Compton.
\end{abstract}

\tableofcontents
\newpage

% =====================================================================
\section{Configuration du nucl\'eon}
% =====================================================================

Le nucl\'eon est un entrelacs irr\'eductible de 3 n\oe uds (P17) :
\[
  L_{\text{nucl\'eon}} = 9_1 + 7_4 + 6_3, \qquad k = 18
\]
avec le nombre d'enroulement de Fibonacci $k=18$, soit $F(18) = """ + str(eq["F_k"]) + r"""$.

\medskip
\noindent
\textbf{Exclusion topologique (P16)} :
le n\oe ud $7_4$ (central) est topologiquement \emph{exclu} : son puits de
potentiel effectif est domin\'e par ceux de $9_1$ et $6_3$.
Seuls $9_1$ et $6_3$ portent la fonction d'onde $\Psi$.
Le nombre de n\oe uds actifs est $N_\Psi = """ + str(N_PSI) + r"""$.

\subsection{\'Equilibre de Cornell (P12)}

Le potentiel de Cornell topologique :
\[
  U(d) = M_{\text{base}} + \frac{A}{d} + \sigma_k\, d
\]
avec les param\`etres du nucl\'eon :

\begin{center}
\begin{tabular}{lrl}
  \toprule
  Quantit\'e & Valeur & Description \\
  \midrule
  $\rho_{\text{total}}$ & $""" + f"{eq['rho_total']:.1f}" + r"""$ & Somme des r\'etractions $\sum_i \rho_J(K_i)$ \\
  $M_{\text{base}}$ & $""" + f"{eq['M_base']:.1f}" + r"""$ & Masse de base $\sum_i M(K_i)$ \\
  $A$ & $""" + f"{eq['A']:.2f}" + r"""$ & Compression quantique \\
  $\sigma_k$ & $""" + f"{eq['sigma_k']:.0f}" + r"""$ & Tension de corde topologique \\
  $d^*$ & $""" + f"{eq['d_star']:.6f}" + r"""$ & Distance inter-n\oe uds \'equilibre \\
  $d^*$ (fm) & $""" + f"{eq['d_star'] * L0:.4f}" + r"""$ & En unit\'es SI \\
  $M_{\text{raw}}$ & $""" + f"{eq['mass_raw']:.2f}" + r"""$ & Masse brute \\
  $m_{\text{nucl\'eon}}$ & $""" + f"{eq['m_nucleon_mev']:.1f}" + r"""~\text{MeV}$ & Masse physique ($\eta \cdot M_{\text{raw}}$) \\
  $E_{\text{liaison}}$ & $""" + f"{eq['E_bind']:.2f}" + r"""$ & \'Energie de liaison Cornell \\
  \bottomrule
\end{tabular}
\end{center}

% =====================================================================
\section{Rayon libre de chaque n\oe ud (partie B)}
% =====================================================================

Le \textbf{rayon libre} d'un n\oe ud isol\'e est le rayon d'exclusion P16 :
\begin{equation}
  \boxed{r_{\text{free}}(K) = \frac{\hbar c}{\rho_J(K)}}
  \label{eq:rfree}
\end{equation}

C'est la r\'esolution minimale o\`u le n\oe ud peut porter une fonction d'onde.
Plus un n\oe ud est topologiquement complexe ($\rho_J$ \'elev\'e), plus son
rayon libre est petit.

\begin{center}
\begin{tabular}{lrrr}
  \toprule
  N\oe ud & $\rho_J$ & $r_{\text{free}}$ (code) & $r_{\text{free}}$ (fm) \\
  \midrule
"""
    for kr in radii_fm:
        tex += f"  ${kr['name'].replace('_', '\\_ ')}$ & ${kr['rho_J']:.1f}$ & ${kr['r_free']:.4f}$ & ${kr['r_free_fm']:.4f}$ \\\\\n"

    tex += r"""  \bottomrule
\end{tabular}
\end{center}

\medskip
\noindent
Interpr\'etation : le n\oe ud $9_1$ (torique T(2,9), le plus complexe) a le
rayon libre le plus petit. Le n\oe ud $6_3$ (amphichiral) est le plus \'etendu.

% =====================================================================
\section{Compression dans le nucl\'eon (partie A)}
% =====================================================================

\subsection{R\'etractation comprim\'ee (P4)}

\`A l'\'energie de liaison $E^* = """ + f"{eq['E_bind']:.0f}" + r"""$ du nucl\'eon,
la r\'etractation de chaque n\oe ud est fortement r\'eduite :
\begin{equation}
  \rho(K_i, E^*) = \frac{\rho_J(K_i)^2}{\rho_J(K_i) + E^*/E_0}
  \label{eq:rhocomp}
\end{equation}

\begin{center}
\begin{tabular}{lrrl}
  \toprule
  N\oe ud & $\rho_J$ & $\rho(K_i, E^*)$ & Ratio $\rho_{\text{comp}}/\rho_J$ \\
  \midrule
"""
    for kr in radii_fm:
        ratio = kr["rho_comp"] / kr["rho_J"]
        tex += f"  ${kr['name'].replace('_', '\\_ ')}$ & ${kr['rho_J']:.1f}$ & ${kr['rho_comp']:.6f}$ & ${ratio:.6f}$ \\\\\n"

    tex += r"""  \bottomrule
\end{tabular}
\end{center}

\noindent
La r\'etractation est r\'eduite d'un facteur $\sim 10^{{-3}}$ : les n\oe uds
sont quasi-dissous \'energ\'etiquement, mais \emph{confin\'es g\'eom\'etriquement}
par le potentiel de Cornell.

\subsection{Rayon de confinement}

Chaque n\oe ud $K_i$ occupe une fraction de l'espace inter-n\oe uds $d^*/2$,
proportionnelle \`a son poids topologique :
\begin{equation}
  \boxed{r_{\text{conf}}(K_i) = \frac{d^*}{2} \cdot \frac{\rho_J(K_i)}{\rho_{\text{total}}}}
  \label{eq:rconf}
\end{equation}

Le \textbf{ratio de compression} $C_i = r_{\text{free}} / r_{\text{conf}}$ mesure
la compression de Fibonacci :

\begin{center}
\begin{tabular}{lrrrr}
  \toprule
  N\oe ud & $r_{\text{free}}$ (fm) & $r_{\text{conf}}$ (fm) & $C_i$ & Actif (P16) \\
  \midrule
"""
    for kr in radii_fm:
        active_str = "oui" if kr["active"] else "non (exclu)"
        tex += f"  ${kr['name'].replace('_', '\\_ ')}$ & ${kr['r_free_fm']:.4f}$ & ${kr['r_conf_fm']:.6f}$ & ${kr['compression_ratio']:.0f}\\times$ & {active_str} \\\\\n"

    tex += r"""  \bottomrule
\end{tabular}
\end{center}

\noindent
Les n\oe uds sont comprim\'es d'un facteur $\sim 135$ \`a $250\times$ par rapport
\`a leur \'etat libre. Cette compression massive est la signature de Fibonacci :
$F(18) = 2584$.

% =====================================================================
\section{Densit\'e particulaire et correction m\'etrique (partie A)}
% =====================================================================

\subsection{Densit\'e particulaire}

La densit\'e de r\'etractation $\rho(K_i, E^*)$ confin\'ee dans le volume
$V_i = \frac{4}{3}\pi\, r_{\text{conf}}^3$ donne la densit\'e particulaire :
\begin{equation}
  n_i = \frac{\rho(K_i, E^*)}{V_i}
  \label{eq:density}
\end{equation}

\subsection{Facteur m\'etrique (P5)}

La m\'etrique \'emergente bi-conforme (P5) :
\[
  \dd s^2 = -c^2 e^{-\kappa|\Psi|^2}\dd t^2
    + \Omega^2(\dd x^2 + \dd y^2 + \dd z^2),
  \qquad \Omega^2 = 1 + \kappa\,|\Psi|^2
\]
Au voisinage du n\oe ud $K_i$, le facteur conforme est :
\begin{equation}
  \Omega_i = \sqrt{1 + \kappa \cdot n_i}
  \label{eq:omega}
\end{equation}

Le \textbf{rayon propre} (distance physique) est :
\begin{equation}
  \boxed{r_{\text{propre}}(K_i) = r_{\text{conf}}(K_i) \times \Omega_i}
  \label{eq:rproper}
\end{equation}

\begin{center}
\begin{tabular}{lrrrr}
  \toprule
  N\oe ud & $n_i$ & $\Omega_i$ & $r_{\text{propre}}$ (code) & $r_{\text{propre}}$ (fm) \\
  \midrule
"""
    for kr in radii_fm:
        tex += (
            f"  ${kr['name'].replace('_', '\\_ ')}$ & "
            f"$\\num{{{kr['n_particle']:.2e}}}$ & "
            f"${kr['Omega']:.1f}$ & "
            f"${kr['r_proper']:.4f}$ & "
            f"${kr['r_proper_fm']:.4f}$ \\\\\n"
        )

    tex += r"""  \bottomrule
\end{tabular}
\end{center}

\medskip
\noindent
\textbf{Interpr\'etation} : la densit\'e particuli\`ere est \'enorme
($\sim 10^7$ en unit\'es code), ce qui produit des facteurs m\'etriques
$\Omega \sim 10^3$. L'espace est consid\'erablement dilat\'e au voisinage
des n\oe uds. Le rayon \emph{propre} (physique) est donc bien plus grand
que le rayon coordonn\'ee, de l'ordre de $\sim 0.6$--$0.7$ fm --- une \'echelle
compatible avec le rayon du proton.

% =====================================================================
\section{Pr\'ediction du rayon du proton (partie C)}
% =====================================================================

\subsection{Formule topologique}

Le rayon de charge du proton est d\'etermin\'e par l'extension radiale de la
fonction d'onde $\Psi$ autour du c\oe ur topologique. Chaque n\oe ud actif
(P16) contribue $2\lambda_C$ \`a l'extension radiale, o\`u $\lambda_C$ est la
longueur d'onde de Compton du nucl\'eon :

\begin{equation}
  \boxed{r_p = 2\,N_\Psi \cdot \frac{\hbar}{m_p c}
    = 2 \times """ + str(N_PSI) + r""" \times \lambda_C
    = 4\,\lambda_C}
  \label{eq:rp}
\end{equation}

\begin{tcolorbox}[colback=blue!5,colframe=blue!50!black,title=D\'erivation du facteur 4]
\begin{enumerate}
  \item Le nucl\'eon contient 3 n\oe uds : $9_1$ ($b=2$), $7_4$ ($b=3$), $6_3$ ($b=4$).
  \item P16 exclut $7_4$ $\Rightarrow$ $N_\Psi = 2$ n\oe uds actifs.
  \item Chaque n\oe ud actif cr\'ee un puits de potentiel $V \propto -\rho_J/r^2$.
  \item L'\'etat li\'e dans un puits $1/r^2$ a une extension radiale
    $\langle r \rangle \approx 2\lambda_C$ (analogue du rayon de Bohr pour
    l'interaction nucléaire forte).
  \item Le rayon de charge est la somme des contributions :
    $r_p = N_\Psi \times 2\lambda_C = 4\lambda_C$.
\end{enumerate}
\end{tcolorbox}

\subsection{R\'esultat num\'erique}

\begin{center}
\begin{tabular}{lrl}
  \toprule
  Quantit\'e & Valeur & \\
  \midrule
  $\lambda_C = \hbar c / (m_p c^2)$ & $""" + f"{rp['lambda_C_fm']:.5f}" + r"""$ & fm \\
  $N_\Psi$ (P16) & $""" + str(rp['N_psi']) + r"""$ & n\oe uds actifs \\
  Facteur & $""" + str(rp['factor']) + r"""$ & $= 2 \times N_\Psi$ \\
  \midrule
  $r_p$ (pr\'edit) & $\mathbf{""" + f"{rp['r_p_fm']:.4f}" + r"""}$ & \textbf{fm} \\
  $r_p$ (exp.)     & $""" + f"{rp['r_p_exp']:.4f}" + r"""$ & fm (CODATA 2018) \\
  Erreur           & $""" + f"{rp['error_pct']:.3f}" + r"""\,\%$ & \\
  \bottomrule
\end{tabular}
\end{center}

% =====================================================================
\section{Rayon du proton vs.\ pression particuli\`ere}
% =====================================================================

\subsection{Mod\`ele}

En faisant varier la distance inter-n\oe uds $d$ autour de l'\'equilibre $d^*$,
on obtient la masse $m(d) = \eta\,U(d)$ et le rayon de charge
$r_p(d) = 4\hbar c / (m(d)\,c^2)$ en fonction de la densit\'e particuli\`ere
\[
  n(d) = \frac{\rho_{\text{total}}}{V(d)}
  = \frac{\rho_{\text{total}}}{\frac{4}{3}\pi (d/2)^3}
\]
et de la pression de Cornell
\begin{equation}
  \boxed{P(d) = -\frac{\dd U}{\dd V}
    = \frac{2}{\pi d^2}\!\left(\frac{A}{d^2} - \sigma_k\right)}
  \label{eq:pressure}
\end{equation}
avec $P > 0$ pour $d < d^*$ (pression de d\'eg\'en\'erescence) et $P = 0$
\`a l'\'equilibre.

\subsection{Limites physiques}

Trois limites bornent le rayon :

\begin{enumerate}
  \item \textbf{Limite de Compton} ($r_p = \lambda_C$) : le rayon atteint la
    longueur d'onde de Compton quand la masse augmente d'un facteur 4 par
    compression ($m \to 4\,m_p$). En-dessous, le proton est dans le
    r\'egime quantique profond.
  \item \textbf{\'Equilibre nucl\'eon} ($r_p = """ + f"{curve['r_p_star']:.4f}" + r"""$~fm) : le
    point physique r\'eel, fix\'e par le minimum de Cornell \`a $d = d^*$.
  \item \textbf{Limite de Schwarzschild} ($r_p = r_S = 2Gm_p/c^2 \approx
    """ + f"{curve['r_schwarzschild']:.2e}" + r"""$~fm) : le proton formerait un
    trou noir. Inaccessible topologiquement car la masse diverge ($1/d$)
    bien avant.
\end{enumerate}

\subsection{Courbe $r_p(n)$ et $r_p(P)$}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{""" + fig_relpath + r"""}
  \caption{Rayon du proton en fonction de la densit\'e particuli\`ere (gauche)
    et de la pression de Cornell (droite). Le point rouge marque l'\'equilibre
    nucl\'eon physique ($d = d^*$). Les lignes horizontales indiquent les
    limites de Compton ($\lambda_C$, orange) et de Schwarzschild ($r_S$, rouge).
    La courbe bleue montre que $r_p \propto 1/m(d) \propto d$ pour la compression
    ($d \ll d^*$) et $r_p \propto 1/d$ pour la tension ($d \gg d^*$).}
  \label{fig:rp_vs_n}
\end{figure}

% =====================================================================
\section{Synth\`ese}
% =====================================================================

\begin{center}
\begin{tabular}{lccccc}
  \toprule
  N\oe ud & $\rho_J$ & $r_{\text{free}}$ (fm) & $r_{\text{conf}}$ (fm) &
    $C_i$ & $r_{\text{propre}}$ (fm) \\
  \midrule
"""
    for kr in radii_fm:
        tex += (
            f"  ${kr['name'].replace('_', '\\_ ')}$ & "
            f"${kr['rho_J']:.1f}$ & "
            f"${kr['r_free_fm']:.4f}$ & "
            f"${kr['r_conf_fm']:.6f}$ & "
            f"${kr['compression_ratio']:.0f}\\times$ & "
            f"${kr['r_proper_fm']:.4f}$ \\\\\n"
        )

    tex += r"""  \bottomrule
\end{tabular}
\end{center}

\bigskip
\noindent\textbf{Conclusions :}
\begin{enumerate}
  \item Les 3 n\oe uds du nucl\'eon sont comprim\'es d'un facteur
    $\sim 135$--$250\times$ par la compression de Fibonacci ($F(18) = 2584$).
  \item La densit\'e particulaire au c\oe ur du nucl\'eon est \'enorme
    ($n \sim 10^7$), ce qui amplifie le facteur m\'etrique $\Omega \sim 10^3$.
  \item Les rayons \emph{propres} (physiques) des n\oe uds sont de l'ordre
    de $0.6$--$0.7$~fm, compatibles avec le rayon du proton.
  \item Le rayon de charge du proton est pr\'edit par la formule
    $r_p = 4\hbar/(m_p c) = """ + f"{rp['r_p_fm']:.4f}" + r"""$~fm, en accord
    \`a $""" + f"{rp['error_pct']:.3f}" + r"""\,\%$ avec la valeur exp\'erimentale.
  \item La courbe $r_p(n)$ montre que le rayon d\'ecro\^it avec la densit\'e
    (compression), born\'e par $\lambda_C$ en r\'egime quantique et par
    $r_S$ en r\'egime gravitationnel.
  \item Le proton physique se situe dans la zone classique, loin des deux
    limites, ce qui confirme la stabilit\'e de l'\'equilibre de Cornell.
\end{enumerate}

\end{document}
"""
    return tex


def run():
    """Point d'entrée : calcule, trace et génère brouillon.pdf."""
    print("=" * 60)
    print("  Rayon des nœuds du nucléon par densité particulaire")
    print("=" * 60)

    # Calcul complet
    result = full_nucleon_analysis()

    # Affichage résumé
    eq = result["cornell"]
    L0 = result["L0_fm"]
    print(f"\n  Nucléon : {'+'.join(NUCLEON_KNOTS)}, k={NUCLEON_K}")
    print(f"  F({NUCLEON_K}) = {eq['F_k']}")
    print(f"  d* = {eq['d_star']:.6f} ({eq['d_star'] * L0:.4f} fm)")
    print(f"  m_nucléon = {eq['m_nucleon_mev']:.1f} MeV")

    print("\n  Rayons par nœud :")
    for kr in result["knot_radii"]:
        active = "✓" if kr["active"] else "✗"
        print(
            f"    {kr['name']} [{active}]: "
            f"r_free={kr['r_free'] * L0:.4f} fm, "
            f"r_conf={kr['r_conf'] * L0:.6f} fm, "
            f"C={kr['compression_ratio']:.0f}×, "
            f"Ω={kr['Omega']:.0f}, "
            f"r_propre={kr['r_proper'] * L0:.4f} fm"
        )

    rp = result["proton_radius"]
    print(f"\n  Rayon du proton :")
    print(f"    Prédit  : {rp['r_p_fm']:.4f} fm")
    print(f"    Expérim.: {rp['r_p_exp']:.4f} fm")
    print(f"    Erreur  : {rp['error_pct']:.3f} %")

    # Courbe r_p vs densité
    print("\n  Calcul de la courbe r_p(n) ...")
    curve = proton_radius_vs_density(n_points=300)

    # Génération figure
    fig_path = os.path.join(_OUTPUT, "rp_vs_density.pdf")
    generate_figure(curve, fig_path)
    print(f"  Figure : {fig_path}")

    # Figure relative path for LaTeX (relative to docs/)
    fig_relpath = os.path.relpath(fig_path, _DOCS)

    # Génération LaTeX
    tex_source = generate_latex(result, curve, fig_relpath)

    tex_path = os.path.join(_DOCS, "brouillon.tex")
    pdf_path = os.path.join(_DOCS, "brouillon.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_source)
    print(f"  LaTeX écrit : {tex_path}")

    # Compilation (2 passes pour les références)
    for pass_num in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", _DOCS, tex_path],
            capture_output=True, text=True, cwd=_DOCS,
        )
        if proc.returncode != 0 and pass_num == 1:
            print(f"  ERREUR pdflatex (pass {pass_num + 1}):")
            for line in proc.stdout.splitlines():
                if line.startswith("!") or "Error" in line:
                    print(f"    {line}")

    if os.path.exists(pdf_path):
        print(f"  PDF généré : {pdf_path}")
    else:
        print("  ÉCHEC de la compilation PDF")

    print(f"\n{'=' * 60}")
    return result


if __name__ == "__main__":
    run()
