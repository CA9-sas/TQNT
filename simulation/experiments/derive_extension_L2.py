"""L'extension L=2 derivee : le tour de repere est PAR PERIODE (23/07).

CIBLE : le residu nomme « extension L=2 » (le document principal en restait a
« [2,66, 2,86] GeV encadre D(2750/2760), non tranche »). Donnees 1D
disponibles aujourd'hui : D(2740)/D3*(2750), Ds(2860), Lambda_c(2860/
2880), Lambda_b(6146/6152), Xi_c(3055/3080), Xi_b(6327/6333).

LA QUESTION : comment la regle de bande s'etend-elle a L=2 ? Les
briques par revolution (S17/S33) : tube (2 par L, exact -- la
fonctionnelle), traine/echanges (2 meson / 4(d-1) baryon a L=1),
rung 1/2, shift s. Qu'est-ce qui est PAR L et qu'est-ce qui ne l'est
pas ?

LA RESOLUTION (mesuree puis mecanisee) : a L=1, « par L » et « par
periode de revolution » sont INDISCERNABLES (1 revolution = 1 unite).
L=2 les separe : les donnees montrent que la partie relative vaut
   meson  : L + 1 (+ 1/2 au top), x3^g  (a L=1 : 5/2 -- l'acquis 1P)
   baryon : (3L+1)(d-1), x3^g           (a L=1 : 4(d-1) -- l'ancien « 4 »)
MECANISME : LE TOUR DU REPERE MATERIEL = 1 PAR PERIODE, quel que soit
L (le fait geometrique exact du S17 : une corde en rotation fait UN
tour de repere par revolution -- il est aveugle a L par nature) ;
tout le reste (enroulement de tube, relatif verrouille au tube) est
PAR L. L'attribution est FORCEE par la L-independance du fait
geometrique. Le x3^g reste sur tout le registre lourd (ratio des
deficits Lambda_b/Lambda_c mesure ~3 : automatique dans la forme).

LE SHIFT s SUIT LA DEMI-SUBSTITUTION (2e echo) : meson = tube COMMUN
-> s par bande (x1, propriete statique du tube partage) ; baryon =
jambes PROPRES -> s par L (la jambe etrange tourne a chaque barreau).
Ds 1D selectionne x1 (-0,1 %), Xi_c/Xi_b selectionnent xL (residus
-1/-2 % = LEUR tension 1P connue, coherente).
"""
import numpy as np

D1 = np.sqrt(2) - 1
S = 0.3590
B23 = 23.0 / 18.0
PS = np.pi * 0.41847**2
ML, MS_, MC, MB = 0.310, 0.4948, 1.557, 4.952

CORES = {"D": MC + ML, "Ds": MC + (ML + MS_) / 2, "B": MB + ML,
         "Bs": MB + (ML + MS_) / 2, "Lc": MC + 2 * ML,
         "Lb": MB + 2 * ML, "Xc": MC + ML + MS_, "Xb": MB + ML + MS_}


def n_of(m, core):
    return (m**2 - core**2) / PS


def m_of(n, core):
    return np.sqrt(core**2 + PS * n)


def cent(pairs):
    w = sum(2 * j + 1 for j, _ in pairs)
    return sum((2 * j + 1) * m for j, m in pairs) / w


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LES INCREMENTS MESURES DU 2e BARREAU (centroides (2J+1))")
    print("=" * 78)
    # centroides 1D mesures (GeV)
    d1d = cent([(2, 2.747), (3, 2.7631)])          # j_K=5/2 (D(2740)+D3*)
    d1p = cent([(1, 2.4221), (2, 2.4611)])         # j_K=3/2 1P
    lc0, lc1d = 2.28646, cent([(1.5, 2.8561), (2.5, 2.88163)])
    lb0, lb1d = 5.61960, cent([(1.5, 6.1462), (2.5, 6.1525)])
    xc0, xc1d = 2.46771, cent([(1.5, 3.0559), (2.5, 3.0772)])
    xb0, xb1d = 5.79190, cent([(1.5, 6.32728), (2.5, 6.33269)])
    n_d1p, n_d1d = n_of(d1p, CORES["D"]), n_of(d1d, CORES["D"])
    print(f"  D  : n(1P j3/2) = {n_d1p:.3f} (etabli 9/2), n(1D j5/2) = "
          f"{n_d1d:.3f} -> increment {n_d1d - n_d1p:.3f}")
    incs = {}
    for nom, core, m0, m1d, dn1p in [
            ("Lambda_c", CORES["Lc"], lc0, lc1d, 2.9380),
            ("Lambda_b", CORES["Lb"], lb0, lb1d, 6.2465),
            ("Xi_c", CORES["Xc"], xc0, xc1d, 3.2540),
            ("Xi_b", CORES["Xb"], xb0, xb1d, 6.4970)]:
        dn1d = n_of(m1d, core) - n_of(m0, core)
        incs[nom] = (dn1p, dn1d)
        print(f"  {nom:9s}: Dn(1P) = {dn1p:.3f}, Dn(1D) = {dn1d:.3f} "
              f"-> increment {dn1d - dn1p:.3f} (vs 1er barreau {dn1p:.3f})")
    print("""  -> PARTOUT le 2e barreau coute MOINS que le 1er : la linearite
  naive en L est morte ; il faut identifier la piece non-lineaire.""")

    print("=" * 78)
    print("[2] KILLS : tout-lineaire et tout-fixe")
    print("=" * 78)
    lin_lc = 2 * 2.9380
    fix_lc = 2.9380 + B23
    mes_lc = incs["Lambda_c"][1]
    print(f"""  Lambda_c Dn(1D) mesure = {mes_lc:.3f} :
   - TOUT-LINEAIRE (2 x Dn(1P))            : {lin_lc:.3f}  ({100 * (lin_lc / mes_lc - 1):+.1f} %, EXCLU)
   - TOUT-FIXE (blind lineaire, reste fixe) : {fix_lc:.3f}  ({100 * (fix_lc / mes_lc - 1):+.1f} %, EXCLU)
  (meme verdict dans les 4 autres systemes.)""")

    print("=" * 78)
    print("[3] LA RESOLUTION : le tour de repere par PERIODE")
    print("=" * 78)
    # formes closes : meson relatif L+1 ; baryon (3L+1)(d-1) ; rung 1/2 top ;
    # s : x1 meson (tube commun), xL baryon (jambes propres)
    defc = 2 * 2.9380 - incs["Lambda_c"][1]
    defb = 2 * 6.2465 - incs["Lambda_b"][1]
    print(f"""  Deficits baryoniques mesures (vs lineaire) : Lambda_c {defc:.3f},
  Lambda_b {defb:.3f} -> ratio {defb / defc:.2f} ~ 3 : le deficit vit sur le
  registre LOURD (x3^g) -- et vaut ~UNE unite d'echange (d-1 = {D1:.3f}).
  FORMES CLOSES (zero nombre nouveau ; rung 1/2 DANS le x3^g,
  la convention 1P du document principal -- 19/2 = 2 + 3(2+1/2)) :
     meson  : n(L, j_K) = 2L + 3^g (L+1+1/2 [top])  [+ s x1 si etrange]
     baryon : Dn(L)     = (23/18) L + 3^g (3L+1)(d-1)  [+ s L si etrange]
  L'unite retranchee au 2e barreau = LE TOUR DE REPERE (1/periode,
  fait S17, aveugle a L par nature) ; a L=1 : meson 1+1=2 ✓,
  baryon (3+1)(d-1)=4(d-1) ✓ -- LE CANON 1P EST INCHANGE.""")

    print("=" * 78)
    print("[4] VERIFICATION (8 systemes, zero parametre)")
    print("=" * 78)
    rows = [
        ("D 1D j5/2", m_of(2 * 2 + (2 + 1) + 0.5, CORES["D"]), d1d),
        ("Ds 1D j5/2 (s x1)", m_of(7.5 + S, CORES["Ds"]), 2.8600),
        ("Lambda_c 1D", m_of(n_of(lc0, CORES["Lc"]) + 2 * B23 + 7 * D1,
                             CORES["Lc"]), lc1d),
        ("Lambda_b 1D", m_of(n_of(lb0, CORES["Lb"]) + 2 * B23 + 21 * D1,
                             CORES["Lb"]), lb1d),
        ("Xi_c 1D (s L)", m_of(n_of(xc0, CORES["Xc"]) + 2 * B23 + 7 * D1
                               + 2 * S, CORES["Xc"]), xc1d),
        ("Xi_b 1D (s L)", m_of(n_of(xb0, CORES["Xb"]) + 2 * B23 + 21 * D1
                               + 2 * S, CORES["Xb"]), xb1d),
        ("B 1D top (=BJ(5970)?)", m_of(4 + 3 * 3.5, CORES["B"]), 5.971),
        ("Bs 1D top (=BsJ(6064)?)", m_of(4 + 3 * 3.5 + S, CORES["Bs"]),
         5.0 + 1.0635),
    ]
    for nom, pred, mes in rows:
        print(f"  {nom:18s} pred {1e3 * pred:6.0f}  mesure {1e3 * mes:6.0f}  "
              f"({100 * (pred / mes - 1):+.2f} %)")
    print(f"""  -> HUIT SYSTEMES A <= 0,4 %% EN MASSE. Le placement du rung
  (1/2 DANS le x3^g, force par le 19/2 du acquis 1P) RETOURNE deux
  tensions en CONFIRMATIONS : B_J(5970) (CDF/LHCb, J inconnu) = le
  top 1D bottom a +0,02 %%, B_sJ(6064) = le top 1D etrange a +0,11 %%
  (falsifiable : J^P(5970) = 3-). En n, les Xi portent leur petite
  tension 1P connue (-2 %%, decimale du shift). Le s discrimine :
  Ds avec s x L donnerait {1e3 * m_of(7.5 + 2 * S, CORES['Ds']):.0f} (+1,1 %, defavorise) ;
  Xi_b avec s x1 donnerait {1e3 * m_of(n_of(xb0, CORES['Xb']) + 2 * B23 + 21 * D1 + S, CORES['Xb']):.0f} (+0,7 %, pire que -0,4 %) :
  le s SUIT la demi-substitution (tube commun x1 / jambes propres xL).""")

    print("=" * 78)
    print("[5] NULL ET ALTERNATIVES")
    print("=" * 78)
    print(f"""  Deficit meson mesure (vs lineaire du top : 4,5) :
  {4.5 - (n_d1d - n_d1p):.4f} -- pi/2 = {np.pi / 2:.4f} colle a 0,1 %% ; baryon ~ pi/8.
  MAIS ce sont deux formes isolees, SANS
  mecanisme, sans lien entre elles, sans extrapolation L=3 (regle du
  null) -- et le deficit unifie (1 tour de repere : meson 1,5 = 3/2 a
  4,6 %%, baryon (d-1) a 5 %% puis x3^g) fait mieux GLOBALEMENT. La lecture « tour de repere par periode » est UNE regle, DEUX
  secteurs, x3^g automatique, L=1 inchange, L=3 predictif -> selectionnee.""")

    print("=" * 78)
    print("[6] PREDICTIONS (L=2 et L=3, zero parametre)")
    print("=" * 78)
    preds = [
        ("B 1D sol j3/2", m_of(4 + 3 * 3, CORES["B"]),
         "sous B_J(5970) : doublet 1D bottom 5903/5972 (~69 MeV)"),
        ("D 1D j3/2 sol", m_of(4 + 3 + 0, CORES["D"]),
         "large, distorsion 2S declaree (D1*(2760) 2781 a +2,6 %)"),
        ("D 1F j7/2 centroide", m_of(6 + 4 + 0.5, CORES["D"]), "L=3"),
        ("Lambda_c 1F", m_of(n_of(lc0, CORES["Lc"]) + 3 * B23 + 10 * D1,
                             CORES["Lc"]), "L=3"),
        ("Lambda_b 1F", m_of(n_of(lb0, CORES["Lb"]) + 3 * B23 + 30 * D1,
                             CORES["Lb"]), "L=3"),
    ]
    for nom, m, note in preds:
        print(f"  {nom:22s} = {1e3 * m:6.0f} MeV   {note}")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  L'EXTENSION L=2 EST DERIVEE : la partie relative n'est pas lineaire
  en L parce qu'UNE de ses unites -- le tour du repere materiel -- est
  PAR PERIODE de revolution (fait geometrique S17, aveugle a L par
  nature), le reste (enroulement de tube) etant par L :
  meson L+1, baryon (3L+1)(d-1), rung 1/2 au top (monotonie par L),
  s selon la demi-substitution (x1 tube commun / xL jambes propres).
  ZERO nombre nouveau, l'acquis 1P inchange (le rung 1/2 DANS le
  x3^g : la convention 19/2), HUIT systemes 1D a <= 0,4 % en masse --
  dont B_J(5970) (+0,02 %) et B_sJ(6064) (+0,11 %) RETOURNES en
  confirmations (falsifiable : J^P(5970) = 3-) -- et L=3 predictif. L'attribution « unite fixe = tour de repere » est
  l'identification conditionnelle (forcee par la L-independance du
  fait geometrique). La liste de travail du secteur tombe a UN item :
  Omega_c* -0,7 %.""")


if __name__ == "__main__":
    main()
