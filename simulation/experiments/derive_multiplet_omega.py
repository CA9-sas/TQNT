"""La regle de multiplet Omega_c/Omega_b (1P, diquark ss spin 1) (22/07).

CONTENU (derive du document principal) : le verrou Pauli force la paire ss en spin 1 ;
en onde P (lambda-mode), les degres legers font j = L x S_ss = 1 x 1 =
{0, 1, 2} ; avec S_Q = 1/2 : J = {1/2}_j0 + {1/2, 3/2}_j1 + {3/2, 5/2}_j2
= CINQ etats. LHCb : quintet Omega_c (2017/2023) ; quartet Omega_b (2020)
-> LE CINQUIEME (j=0, J=1/2) MANQUE : a predire.

PIECES DERIVEES APPLIQUEES :
 - monotonie (S28) : j=0 (anti-aligne) = SOL de l'echelle d'alignement ;
 - splitting lourd (S27) : E/unite <S_Q.j> = h_{1/2} sigma_chi/m_Q,
   dilution PLEINE ss (jambes propres, baryon) : x (m_l/m_s) par etrange
   -> ici les DEUX jambes sont etranges : x m_l/m_s ;
 - l'echelle d'alignement de la paire (L.S_ss) est un terme LEGER :
   HEAVY-BLIND (meme pas en n de c a b) -- teste, puis transfere pour
   predire le 5e etat.

Hypothese d'assignation (charme, ordre LHCb 2023) : 3000 (j0, J=1/2) ;
3050/3065 (j1 : J=1/2, 3/2) ; 3090/3119 (j2 : J=3/2, 5/2). Bottom
(spins NON mesures) : appariement adjacent 6316/6330 (j1), 6340/6350
(j2) -- teste contre l'alternative.
"""
import numpy as np

SIGMA = 0.41847 ** 2
PS = np.pi * SIGMA
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
H12 = 3 / 16.0
DIL = ML / MS                        # dilution pleine (jambes ss)

OC_CORE, OB_CORE = MC + 2 * MS, MB + 2 * MS
OC = {"g": 2.6952, "j0": [(3.0004, 0.5)],
      "j1": [(3.0502, 0.5), (3.0655, 1.5)],
      "j2": [(3.0901, 1.5), (3.1191, 2.5)]}
OB = {"g": 6.0458,
      "j1": [(6.31564, 0.5), (6.33030, 1.5)],
      "j2": [(6.33971, 1.5), (6.34988, 2.5)]}


def nv(m, core):
    return (m * m - core * core) / PS


def cen_n(states, core):
    w = np.array([2 * j + 1 for _, j in states])
    return (w * np.array([nv(m, core) for m, _ in states])).sum() / w.sum()


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] L'ECHELLE D'ALIGNEMENT MESUREE (en n, niveaux j)")
    print("=" * 78)
    nc = {k: cen_n(v, OC_CORE) for k, v in OC.items() if k != "g"}
    nb = {k: cen_n(v, OB_CORE) for k, v in OB.items() if k != "g"}
    sc01 = nc["j1"] - nc["j0"]
    sc12 = nc["j2"] - nc["j1"]
    sb12 = nb["j2"] - nb["j1"]
    print("""  charme : n(j0) = %.3f, n(j1) = %.3f, n(j2) = %.3f
           pas j0->j1 = %.3f ; pas j1->j2 = %.3f
  bottom : n(j1) = %.3f, n(j2) = %.3f ; pas j1->j2 = %.3f
  -> HEAVY-BLINDNESS du pas j1->j2 : c %.3f vs b %.3f (%+.0f %%) --
  l'echelle d'alignement est bien un terme LEGER (comme le pas des
  bandes mesoniques) a ~10 %% pres (contamination 1/m). Candidat de
  forme : ~1/2 par niveau j (le pas j1->j2 a +6/-6 %% du demi-tour) ;
  le pas j0->j1 (%.3f, +32 %%) s'en ecarte -- SOL non uniforme ou
  assignation 3000 (2S ?) suspecte : NON PINCE (regle du null).""" % (
        nc["j0"], nc["j1"], nc["j2"], sc01, sc12,
        nb["j1"], nb["j2"], sb12, sc12, sb12, 100 * (sb12 / sc12 - 1),
        sc01))

    print("=" * 78)
    print("[2] LE CINQUIEME ETAT Omega_b PREDIT (j=0, J=1/2)")
    print("=" * 78)
    n_j0_transfer = nb["j1"] - sc01          # pas charme transfere (blind)
    n_j0_half = nb["j1"] - 0.5               # variante 1/2-par-niveau
    m_t = np.sqrt(OB_CORE**2 + PS * n_j0_transfer)
    m_h = np.sqrt(OB_CORE**2 + PS * n_j0_half)
    print("""  transfert du pas charme (heavy-blind) : n = %.3f -> M = %.0f MeV
  variante pas = 1/2 exact               : n = %.3f -> M = %.0f MeV
  => PREDICTION : Omega_b(1P, j=0, J=1/2) = %.0f-%.0f MeV, sous le
  quartet 2020 (largeur attendue superieure -- j=0 s'annihile plus
  volontiers en Xi_b K en onde S ; c'est plausiblement POURQUOI il
  manque au quartet).""" % (
        n_j0_transfer, 1e3 * m_t, n_j0_half, 1e3 * m_h,
        1e3 * min(m_t, m_h), 1e3 * max(m_t, m_h)))

    print("=" * 78)
    print("[3] SPLITTINGS LOURDS INTERNES (S27 : h_{1/2} + dilution pleine ss)")
    print("=" * 78)
    for nom, core, mq, dd in (("Omega_c", OC_CORE, MC, OC),
                              ("Omega_b", OB_CORE, MB, OB)):
        unit = H12 * SIGMA / mq * DIL
        # <S_Q.j> : j=1 (J=3/2 vs 1/2) : +1/2 vs -1 -> 3/2 unites ;
        #           j=2 (J=5/2 vs 3/2) : +1  vs -3/2 -> 5/2 unites
        p1, p2 = 1.5 * unit, 2.5 * unit
        o1 = dd["j1"][1][0] - dd["j1"][0][0]
        o2 = dd["j2"][1][0] - dd["j2"][0][0]
        print("  %-8s j1 : obs %5.1f vs pred %5.1f MeV (%+.0f %%) ; "
              "j2 : obs %5.1f vs pred %5.1f (%+.0f %%)" % (
                  nom, 1e3 * o1, 1e3 * p1, 100 * (o1 / p1 - 1),
                  1e3 * o2, 1e3 * p2, 100 * (o2 / p2 - 1)))
    print("""  -> charme : ratio j2/j1 obs 1,90 vs Clebsch 5/3 = 1,67 (+14 %) OK ;
  amplitudes -23/-12 % (assignations 2023, dilution pleine) : SUPPORTE.
  bottom : j2 a -2 %% (!) mais j1 obs 14,7 vs 6,2 pred -- INVERSION du
  ratio. CANDIDAT DECLARE : les deux J=3/2 (6330,3/6339,7 -- a 9,4 MeV
  l'un de l'autre !) se MELANGENT a O(1/m_b) (memes nombres quantiques
  exacts) : la repulsion ecarte 6330 vers le bas et 6340 vers le haut,
  gonflant le splitting apparent de j1 et preservant j2 -- PREDICTION
  jointe : les deux Omega_b(3/2) sont fortement melanges (testable par
  leurs rapports de branchement Xi_b K / Xi_b' K) ; les spins du
  quartet (non mesures) trancheront.""")

    print("=" * 78)
    print("[4] LE CENTROIDE DU MULTIPLET : PIECE OUVERTE (declare)")
    print("=" * 78)
    dn_c = (1 * nc["j0"] + 3 * nc["j1"] + 5 * nc["j2"]) / 9 - nv(OC["g"], OC_CORE)
    dn_b_q = (3 * nb["j1"] + 5 * nb["j2"]) / 8 - nv(OB["g"], OB_CORE)
    print("""  Dn(Omega_c, quintet complet) = %.3f ; Dn(Xi_c) = 3,254 :
  exces +0,79. Dn(Omega_b, quartet observe) = %.3f ; Dn(Xi_b) = 6,497 :
  exces ~0,0. AUCUNE decomposition simple (2e shift s ? centroide
  d'alignement ?) ne tient AUX DEUX generations a la fois : la piece
  centroide de la regle de multiplet reste OUVERTE -- suspects nommes :
  contenu du quintet charme (3000 ou 3119 = 2S ?), membre j0 bottom
  manquant, melange 3/2. Regle du null appliquee : rien de retenu.""" % (
        dn_c, dn_b_q))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  DERIVE : la regle de CONTENU (verrou ss spin 1 x L=1 -> j=0,1,2 ->
  5 etats ; le sol j=0 par monotonie) ; SUPPORTE : heavy-blindness de
  l'echelle d'alignement (~10 %), splittings lourds (charme ratio +14 %,
  bottom j2 -2 %%) sous h_{1/2} + dilution pleine ; PREDIT :
  **Omega_b(1P, j=0, J=1/2) = 6297-6304 MeV** (le cinquieme du quartet
  2020) + melange fort des deux Omega_b(3/2) ; OUVERT : le pas j0->j1
  (0,66 vs 1/2), le centroide (incoherent c/b sous les decompositions
  simples), les spins du quartet bottom. La regle de multiplet est
  DEMI-DERIVEE : contenu + sol + splittings oui, echelle d'alignement
  mesuree mais non pincee.""")


if __name__ == "__main__":
    main()
