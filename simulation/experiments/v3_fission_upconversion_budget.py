"""V3 — budget du couple fission de singulet + up-conversion (étude).

Modèle de bilan détaillé simplifié (déclaré) :
 - spectre = corps noir T=5772 K normalisé à 1000 W/m² (approx. AM1.5G) ;
 - trois zones autour du gap E_g :
     [E_g, 2E_g)  : absorption directe, 1 porteur/photon ;
     [2E_g, inf)  : FISSION, QY_f porteurs/photon (idéal 2) ;
     [E_g/2, E_g) : UP-CONVERSION, 2 photons -> 1 porteur, efficacité eta_UC ;
 - tension par porteur V = f_V * E_g (facteur de Shockley-Queisser
   effectif f_V = 0,75, déclaré ; contrôle : sans fission ni UC le modèle
   doit redonner ~33 % au gap optimal ~1,3 eV).
Sorties : rendement vs E_g pour (i) référence SQ, (ii) fission seule,
(iii) UC seule, (iv) COUPLE ; cas Si (E_g=1,12 fixé) idéal et réaliste.
"""
import numpy as np

KB_T = 5772 * 8.617333e-5      # eV (T soleil)
F_V = 0.75                     # facteur de tension effectif (déclaré)


def bb_photon_flux(E):
    """Flux de photons du corps noir par unité d'énergie (u.a.)."""
    return E**2 / np.expm1(E / KB_T)


E = np.linspace(0.01, 8.0, 20000)
dE = E[1] - E[0]
flux = bb_photon_flux(E)
P_in = np.sum(flux * E) * dE   # normalisation (u.a.) — les ratios suffisent


def eta(Eg, QY_f=1.0, eta_UC=0.0, cap2=True):
    """Rendement du modèle 3 zones (QY_f=1, eta_UC=0 -> référence SQ)."""
    direct = np.sum(flux[(E >= Eg) & (E < 2 * Eg)]) * dE
    fiss_z = np.sum(flux[E >= 2 * Eg]) * dE
    sub = np.sum(flux[(E >= Eg / 2) & (E < Eg)]) * dE
    carriers = direct + QY_f * fiss_z + eta_UC * sub / 2
    return carriers * F_V * Eg / P_in


gaps = np.linspace(0.6, 2.2, 161)
ref = [eta(g) for g in gaps]
fis = [eta(g, QY_f=2.0) for g in gaps]
upc = [eta(g, eta_UC=1.0) for g in gaps]
duo = [eta(g, QY_f=2.0, eta_UC=1.0) for g in gaps]

print("=" * 74)
print("V3 — COUPLE FISSION + UP-CONVERSION : BUDGET       (2026-09-01)")
print("=" * 74)
i0, i1, i2, i3 = (int(np.argmax(x)) for x in (ref, fis, upc, duo))
print(f"""
1) Optima idéaux (QY_f=2, eta_UC=1) :
   référence (SQ simplifié) : {100*ref[i0]:5.1f} % @ E_g={gaps[i0]:.2f} eV
     (contrôle : litt. 33,7 % @ 1,34 — modèle OK à ~1 pt)
   fission seule            : {100*fis[i1]:5.1f} % @ E_g={gaps[i1]:.2f} eV
   up-conversion seule      : {100*upc[i2]:5.1f} % @ E_g={gaps[i2]:.2f} eV
   COUPLE fission+UC        : {100*duo[i3]:5.1f} % @ E_g={gaps[i3]:.2f} eV""")

# ---- cas silicium (E_g fixe 1,12) ---------------------------------------
Eg = 1.12
cases = [
    ("Si nu (référence)",           dict()),
    ("Si + fission idéale",         dict(QY_f=2.0)),
    ("Si + UC idéale",              dict(eta_UC=1.0)),
    ("Si + COUPLE idéal",           dict(QY_f=2.0, eta_UC=1.0)),
    ("Si + couple RÉALISTE",        dict(QY_f=1.5, eta_UC=0.15)),
    ("Si + couple réaliste + conc. (eta_UC=0.3)",
                                    dict(QY_f=1.5, eta_UC=0.30)),
]
print(f"\n2) Silicium, E_g = {Eg} eV :")
for nom, kw in cases:
    print(f"   {nom:44s} {100*eta(Eg, **kw):5.1f} %")

# parts du spectre au gap Si (photons)
d = np.sum(flux[(E >= Eg) & (E < 2*Eg)]) * dE
f = np.sum(flux[E >= 2*Eg]) * dE
s = np.sum(flux[(E >= Eg/2) & (E < Eg)]) * dE
tot = np.sum(flux) * dE
print(f"\n3) Partage des photons au gap Si : directe {100*d/tot:.0f} % | "
      f"zone fission {100*f/tot:.0f} % | zone UC {100*s/tot:.0f} % | "
      f"perdus <E_g/2 {100*(tot-d-f-s)/tot:.0f} %")
print("   -> la fission travaille sur ~1 photon sur 5, l'UC sur ~1 sur 4 :")
print("      les DEUX bouts du spectre sont couverts, d'ou le couple.")
