"""Théorie Nodale — Masse et énergie du neutrino en fonction du rayon R

Le neutrino est une onde sans masse propre tournant à v = c sur un rayon R.
La quantification du spin (L = ℏ/2) impose :

    m_ν = ℏ / (2Rc)      [masse par inertie rotationnelle]
    E_ν = m_ν c² = ℏc / (2R)   [énergie rotationnelle]

R dépend du niveau de cassure de la longueur d'onde du nœud parent.
Ce script trace m(R) et E(R) de R = 6 fm (cassure nucléaire) à R = 2.5 μm
(limite cosmologique Planck).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.neutrino import HBAR_C_MEV_FM, mass_from_radius, boomerang_radius


# --- Plage de rayons ---
R_min_fm = 6.0           # 6 fm (cassure nucléaire profonde)
R_max_fm = 2.5e9         # 2.5 μm en fm

R_fm = np.geomspace(R_min_fm, R_max_fm, 1000)

# --- Calculs ---
mass_mev = np.array([mass_from_radius(r) for r in R_fm])
energy_mev = mass_mev  # E = mc² (en unités naturelles où c=1, E en MeV = m en MeV)

# Conversions pour affichage
mass_ev = mass_mev * 1e6   # MeV → eV

# --- Limites expérimentales ---
exp_limits = {
    r"$\nu_\tau$ directe (18.2 MeV)": {"m_mev": 18.2, "color": "red"},
    r"$\nu_\mu$ directe (0.17 MeV)": {"m_mev": 0.17, "color": "orange"},
    r"$\nu_e$ KATRIN (0.45 eV)": {"m_mev": 0.45e-6, "color": "green"},
    r"Planck $\Sigma m_\nu/3$ (0.04 eV)": {"m_mev": 0.04e-6, "color": "blue"},
}

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    r"Neutrino : masse et énergie par inertie rotationnelle"
    "\n"
    r"$m_\nu = \hbar / (2Rc)$, $E_\nu = \hbar c / (2R)$"
    "  —  R = niveau de cassure de la longueur d'onde",
    fontsize=13,
)

# --- Axe 1 : Masse ---
ax1.loglog(R_fm, mass_mev, "k-", linewidth=2, label=r"$m_\nu(R) = \hbar c / (2R)$")

for label, info in exp_limits.items():
    m = info["m_mev"]
    R_exp = boomerang_radius(m)
    ax1.axhline(m, color=info["color"], linestyle="--", alpha=0.6, linewidth=1)
    ax1.plot(R_exp, m, "o", color=info["color"], markersize=8, zorder=5)
    # Placer le texte à gauche ou droite selon la position
    ax1.annotate(
        label,
        xy=(R_exp, m),
        xytext=(15, 5),
        textcoords="offset points",
        fontsize=8,
        color=info["color"],
        fontweight="bold",
    )

ax1.set_ylabel("Masse (MeV)", fontsize=12)
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc="upper right", fontsize=10)

# Axe secondaire en eV
ax1_ev = ax1.twinx()
y1, y2 = ax1.get_ylim()
ax1_ev.set_ylim(y1 * 1e6, y2 * 1e6)
ax1_ev.set_yscale("log")
ax1_ev.set_ylabel("Masse (eV)", fontsize=12)

# --- Axe 2 : Énergie ---
ax2.loglog(R_fm, energy_mev, "k-", linewidth=2, label=r"$E_\nu(R) = \hbar c / (2R)$")

for label, info in exp_limits.items():
    m = info["m_mev"]
    R_exp = boomerang_radius(m)
    ax2.axhline(m, color=info["color"], linestyle="--", alpha=0.6, linewidth=1)
    ax2.plot(R_exp, m, "o", color=info["color"], markersize=8, zorder=5)

ax2.set_xlabel("Rayon de rotation R (fm)", fontsize=12)
ax2.set_ylabel(r"Énergie $E = m_\nu c^2$ (MeV)", fontsize=12)
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="upper right", fontsize=10)

# Axe secondaire en unités lisibles pour R
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
ax2_top.set_xscale("log")
# Placer des ticks en pm, nm, μm
tick_positions = [1e1, 1e3, 1e5, 1e7, 1e9]
tick_labels = ["10 fm", "1 pm", "100 pm", "10 nm", r"1 $\mu$m"]
ax2_top.set_xticks(tick_positions)
ax2_top.set_xticklabels(tick_labels, fontsize=9)

# --- Annotations des régimes ---
ax1.axvspan(R_min_fm, 1e2, alpha=0.08, color="red", label="Nucléaire")
ax1.axvspan(1e2, 1e6, alpha=0.08, color="orange", label="Sub-atomique")
ax1.axvspan(1e6, 1e8, alpha=0.08, color="green", label="Moléculaire")
ax1.axvspan(1e8, R_max_fm, alpha=0.08, color="blue", label="Microscopique")

ax1.text(3e1, mass_mev.min() * 5, "nucléaire", fontsize=9, color="red", alpha=0.7)
ax1.text(1e4, mass_mev.min() * 5, "sub-atomique", fontsize=9, color="orange", alpha=0.7)
ax1.text(3e6, mass_mev.min() * 5, "moléculaire", fontsize=9, color="green", alpha=0.7)
ax1.text(3e8, mass_mev.min() * 5, "microscopique", fontsize=9, color="blue", alpha=0.7)

plt.tight_layout()
save_path = "output/neutrino_radius_mass_energy.png"
fig.savefig(save_path, dpi=200, bbox_inches="tight")
print(f"Figure sauvegardée : {save_path}")

# --- Quelques valeurs numériques ---
print("\n=== Valeurs clés ===")
print(f"{'R':<15} {'m (MeV)':<18} {'m (eV)':<18} {'E (MeV)':<18}")
print("-" * 70)
for r_val, label in [
    (6.0, "6 fm"),
    (100.0, "100 fm"),
    (1e4, "10 pm"),
    (1e6, "1 nm"),
    (1e8, "100 nm"),
    (2.5e9, "2.5 μm"),
]:
    m = mass_from_radius(r_val)
    print(f"{label:<15} {m:<18.6e} {m*1e6:<18.6e} {m:<18.6e}")
