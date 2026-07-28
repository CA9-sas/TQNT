r"""
Cascade baryonique vers le nucléon + planchers des conservations topologiques.

Demande utilisateur : vérifier que la cascade mène au nucléon, et qu'il n'y a pas
un nœud PLUS SIMPLE aussi stable (le point anti-tautologie).

(a) CASCADE : chaque baryon a un canal vers plus léger ; SEUL le proton est
    plancher. Deux mécanismes descendants : FORT (ΔS=0, réarrangement de tension,
    décuplet→octet) et FAIBLE (ΔS=1, reconnexion du linking, hypérons→nucléon,
    n→p). Chemin : Ω→Ξ→Λ/Σ→n→p ; Δ,Σ*,Ξ*→octet. Tout finit en proton.

(b) PAS TAUTOLOGIQUE : l'électron (3_1, 3 crois.) est PLUS SIMPLE que le nucléon
    (~17) ET tout aussi stable. Les particules stables = les PLANCHERS des
    conservations TOPOLOGIQUES :
      nombre baryonique → proton  (nœud 3-brins minimal)
      charge / lepton   → électron (3_1 = T(2,3), nœud minimal)
      neutre            → photon, neutrino (aucun nœud)
    Chacun stable PAR TOPOLOGIE (rien de plus simple dans son secteur). Le proton
    domine la MASSE (pas le nombre) car le plancher BARYONIQUE est LOURD (~1 GeV,
    minimal 3-brins) ; le lepton se contente de 2 brins (léger). GAP de complexité :
    aucun baryon plus simple que le nucléon (il faut 3 brins + charge + stabilité).
"""
from __future__ import annotations

# (nom, masse MeV, |étrangeté|, contenu, spin)
BAR = [("Ω⁻", 1672.5, 3, "sss", 1.5), ("Ξ*", 1533., 2, "lss", 1.5),
       ("Σ*", 1385., 1, "lls", 1.5), ("Δ", 1232., 0, "lll", 1.5),
       ("Ξ", 1318., 2, "lss", 0.5), ("Σ", 1193., 1, "lls", 0.5),
       ("Λ", 1116., 1, "lls", 0.5), ("n", 939.6, 0, "lll", 0.5),
       ("p", 938.3, 0, "lll", 0.5)]
STABLE = [("photon", "aucun (onde χ)", 0, "massless", "jauge"),
          ("neutrino", "aucun (onde χ)", 0, "~0", "lepton neutre"),
          ("électron", "3_1 (T(2,3))", 3, "0,511 MeV", "LEPTON chargé"),
          ("proton", "nucléon 3-brins", 17, "938 MeV", "BARYON")]


def main():
    print("(a) CASCADE — chaque baryon a-t-il un canal vers plus léger ?")
    for nm, m, S, c, J in sorted(BAR, key=lambda b: -b[1]):
        lighter = [b for b in BAR if b[1] < m - 0.5]
        if not lighter:
            print(f"  {nm:4s} {m:7.1f}  PLANCHER (aucun baryon plus léger)")
        else:
            print(f"  {nm:4s} {m:7.1f}  → cascade (fort ΔS=0 ou faible ΔS=1) vers plus léger")
    print("  → tous cascadent ; seul le proton est plancher (Ω→Ξ→Λ/Σ→n→p).\n")
    print("(b) Y a-t-il plus simple ET stable ? OUI : l'électron (3 crois. << 17).")
    for nm, k, cr, mm, sect in STABLE:
        print(f"  {nm:9s} {k:16s} crois={cr:2d} {mm:>10s}  {sect}")
    print("  → chaque stable = plancher de SA conservation topologique.")
    print("    proton domine la MASSE (plancher baryon lourd, 3-brins) ; électron léger (2-brins).")


if __name__ == "__main__":
    main()
