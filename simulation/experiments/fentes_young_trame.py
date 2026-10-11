"""Fentes de Young et trame de l'espace-temps — les trois canaux (Y1-Y3).

Hypothèse utilisateur (2026-09-01) : la fente altère la trame par
COMPENSATION de l'écrasement de la matière de l'écran, et la dispersion
en longueur d'onde se fait à ce niveau.

Y1 — canal ÉCRASEMENT (métrique) quantifié : écran de cuivre t=1 mm,
     fente a=100 um. Feuille infinie : g = 2 pi G sigma_s ;
     n-1 = DeltaPhi/c^2 ; déphasage à travers la fente vs pi requis.
Y2 — canal COMPENSATION EM (extinction d'Ewald-Oseen) : les courants de
     framing des noeuds de l'écran annulent l'onde dans le métal sur
     l'épaisseur de peau delta ~ c/omega_p (cuivre : lambda_p ~ 115 nm).
Y3 — la dispersion en lambda : dispersion nulle de la trame (tous les
     lambda à c) -> le tri est ANGULAIRE et GÉOMÉTRIQUE : Fraunhofer
     numérique (FFT) pour 2 lambdas, theta = lambda/a vérifié ; réseau
     (routeur V3-b) : sin theta = lambda/d.
"""
import numpy as np

G = 6.6710e-11        # convention du cadre (G_prédit)
C = 2.99792458e8

print("=" * 74)
print("FENTES DE YOUNG / TRAME — LES TROIS CANAUX          (2026-09-01)")
print("=" * 74)

# ---- Y1 : canal écrasement ----------------------------------------------
rho_cu, t, a, lam = 8960.0, 1e-3, 100e-6, 500e-9
sigma_s = rho_cu * t                     # kg/m^2
g_sheet = 2 * np.pi * G * sigma_s        # m/s^2 (feuille infinie)
dPhi = g_sheet * a                       # variation de potentiel sur la fente
n_minus_1 = dPhi / C**2
dphase = (2 * np.pi / lam) * n_minus_1 * a   # chemin d'interaction ~ a
print(f"""
Y1 — écrasement de l'écran (Cu, t=1 mm, fente a=100 um, lambda=500 nm) :
   sigma_s = {sigma_s:.1f} kg/m^2 ; g_feuille = {g_sheet:.2e} m/s^2
   n-1 (métrique) = {n_minus_1:.2e}
   déphasage à travers la fente = {dphase:.2e} rad  vs  pi requis
   -> DÉFICIT x{np.pi/dphase:.1e} : le canal écrasement NE PEUT PAS
      être le mécanisme de la diffraction (réfuté par ~27 ordres).""")

# ---- Y2 : canal extinction EM -------------------------------------------
lam_p = 115e-9                            # longueur d'onde plasma du cuivre
delta = lam_p / (2 * np.pi)
print(f"""
Y2 — la vraie compensation : extinction EM (Ewald-Oseen)
   Les courants de framing des noeuds de l'écran rayonnent en opposition
   de phase et ANNULENT l'onde dans le métal sur l'épaisseur de peau :
   delta ~ lambda_p/2pi = {delta*1e9:.0f} nm  << a = {a*1e6:.0f} um
   -> bord de fente NET à l'échelle de l'onde ; couplage au niveau alpha
      (EM), ~30 ordres au-dessus du canal métrique : c'est LUI qui fait
      la condition aux limites.""")

# ---- Y3 : dispersion géométrique (Fraunhofer numérique) -----------------
N = 2**16
x = np.linspace(-2e-3, 2e-3, N)
dx = x[1] - x[0]
ouverture = (np.abs(x) < a / 2).astype(float)
print("\nY3 — le tri en lambda est angulaire et géométrique (FFT 1 fente) :")
for lam_i, nom in [(400e-9, "bleu 400 nm"), (700e-9, "rouge 700 nm")]:
    champ = np.fft.fftshift(np.fft.fft(ouverture))
    fx = np.fft.fftshift(np.fft.fftfreq(N, dx))     # fréquences spatiales
    theta = lam_i * fx                              # petits angles
    I = np.abs(champ)**2
    # premier zéro numérique (au-delà du centre)
    i0 = N // 2
    izero = i0 + np.argmax(I[i0:] < I[i0] * 1e-6)
    th_num = theta[izero]
    print(f"   {nom:14s} : 1er zéro numérique = {1e3*th_num:.3f} mrad "
          f"vs lambda/a = {1e3*lam_i/a:.3f} mrad "
          f"({100*(th_num/(lam_i/a)-1):+.2f} %)")
print("""   MÊME vitesse c pour les deux (dispersion nulle de la trame,
   théorème) ; la séparation naît du GRADIENT DE PHASE sur l'ouverture
   (theta = lambda/a), pas d'une altération locale de la trame.""")

# réseau = fentes de Young généralisées -> routeur V3-b
d_res = 1.0e-6
print("\n   Réseau (routeur V3-b), pas d = 1 um, ordre 1 :")
for lam_i, nom in [(400e-9, "bleu"), (550e-9, "vert"), (700e-9, "rouge"),
                   (1000e-9, "IR->UC")]:
    print(f"     {nom:7s} sin(theta)=lambda/d -> "
          f"theta = {np.degrees(np.arcsin(lam_i/d_res)):.1f} deg")
print("   -> le tri 3 voies de la V3 sort de la géométrie seule.")
