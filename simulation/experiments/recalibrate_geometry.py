r"""
Recalibrer l'ancienne formule de masse (omega_unified, U_κ/U_τ) avec la VRAIE
géométrie — test décisif demandé par l'utilisateur.

Deux géométries correctes essayées :
  1. Courbes RÉPARÉES par tresse (Alexander) : INUTILISABLES pour U_τ. Brutes →
     artefacts de coin (U_κ~1500) ; lissées → torsion perdue (U_τ=0, G=ΣU_κU_τ=0).
     La torsion des tresses est aux arcs de fermeture, pas aux vrais croisements.
  2. Table Abraham (REFERENCE_N50K, plongements IDÉAUX à N=50k, torsion correcte) :
     U_κ=(∮κ)²/L, U_τ=(∮|τ|)²/(fL). C'est le test VALIDE ci-dessous.

RÉSULTAT (géométrie idéale correcte) : RMS = 82 % (vs 2 % pour l'artefact des
courbes fausses). ÉCHEC STRUCTUREL : hadrons légers +~100 % (trop lourds), secteur
EW -~70 % (trop légers) — décalages OPPOSÉS. Raison : U_κ(8₃, cœur EW) s'effondre
de 1465 (artefact d'unknot) à 18 (réel), ×80 → le rapport géométrique EW/baryon
change du tout au tout ; aucune calibration unique ne réconcilie les deux.

CONCLUSION : une formule de masse ABSOLUE via la géométrie U_κ/U_τ n'est PAS
récupérable par recalibration. Le 0,7-2 % de l'ancienne formule était ENTIÈREMENT
l'artefact des courbes fausses (calibration per-nœud). Confirme définitivement le
fil « courbes invalides ». Ce qui survit = les prédictions STRUCTURELLES (Koide
61 ppm, Regge 1,5 %) ; le spectre absolu est non-perturbatif (comme la QCD).
"""
import sys; sys.path.insert(0,'.')
import numpy as np
import theory.omega_ts as ots
from theory.abraham import REFERENCE_N50K as ABRAHAM_TABLE

# extraire les données de plongement idéal (L, ∮κ, ∮|τ|)
geo={k:v for k,v in ABRAHAM_TABLE.items() if isinstance(v,dict) and 'L' in v}
def Uk_Ut(K,f=0.5):
    g=geo[K]; L=g['L']
    Uk=g['int_kappa']**2/L          # (∮κ)²/L (Cauchy-Schwarz, validé)
    Ut=g['int_abs_tau']**2/(f*L)     # (∮|τ|)²/(fL)
    return Uk,Ut

need=['3_1','5_2','6_2','6_3','7_1','7_2','8_1','8_3','9_1','10_1']
have=[k for k in need if k in geo]
print("Abraham (plongement idéal) disponible pour :",have)
print("manquants :",[k for k in need if k not in geo])
NEW={k:Uk_Ut(k) for k in have}
print("\nU_κ, U_τ (Abraham idéal) vs artefact catalogue :")
for k in have:
    old=ots._axiom_data(k); print(f"  {k:5s} U_κ:{NEW[k][0]:6.1f} (cat {old['Uk']:7.1f})  U_τ:{NEW[k][1]:7.1f} (cat {old['Ut']:7.1f})")

_orig=ots._axiom_data
def patched(name):
    d=dict(_orig(name))
    if name in NEW: d['Uk'],d['Ut']=NEW[name]
    return d
ots._axiom_data=patched
import importlib, simulation.experiments.omega_unified as ou; importlib.reload(ou)

# particules dont TOUS les composants sont dans have
PART={"p":("3_1+7_2+7_2",938.3),"Λ":("3_1+5_2+7_2",1115.7),"Σ":("5_2+7_2+7_2",1193.2),
 "Ξ":("3_1+3_1+7_2",1318.3),"Δ":("5_2+6_3",1232.0),"ρ⁰":("6_2+6_2",775.3),"ω":("8_1+8_1",782.7),
 "W":("7_1+7_2+8_3",80379.),"Z":("7_2+8_3+9_1",91188.),"H":("7_1+8_3+10_1",125250.)}
def rms_for_x(x):
    L=[]
    for nm,(cfg,mx) in PART.items():
        try: f=ou.mass_unified(cfg,x)
        except Exception: return 9e9,None
        if not np.isfinite(f) or f<=0: return 9e9,None
        L.append((nm,np.log(f),mx))
    off=np.mean([np.log(mx)-a for _,a,mx in L])
    res={nm:(np.exp(a+off),mx) for nm,a,mx in L}
    rms=np.sqrt(np.mean([((mp-mx)/mx)**2 for mp,mx in res.values()]))*100
    return rms,res
from scipy.optimize import minimize_scalar
b=minimize_scalar(lambda x:rms_for_x(x)[0],bounds=(-6,3),method='bounded')
rms,res=rms_for_x(b.x)
print(f"\n=== RECALIBRATION (Abraham idéal, x={b.x:.2f}, 1 échelle) ===")
print(f"  {'part.':5s} {'M_préd':>9s} {'M_exp':>9s} {'écart%':>8s}")
for nm,(mp,mx) in res.items(): print(f"  {nm:5s} {mp:9.0f} {mx:9.0f} {(mp-mx)/mx*100:+8.0f}")
print(f"\n  RMS = {rms:.0f}%  (recalibré, géométrie idéale correcte)")
# baryons seuls vs EW seuls
bar=[nm for nm in ['p','Λ','Σ','Ξ','Δ','ρ⁰','ω']]; ew=['W','Z','H']
rb=np.sqrt(np.mean([((res[n][0]-res[n][1])/res[n][1])**2 for n in bar]))*100
re=np.sqrt(np.mean([((res[n][0]-res[n][1])/res[n][1])**2 for n in ew]))*100
print(f"  -> hadrons légers : {rb:.0f}%   secteur EW : {re:.0f}%  (décalages OPPOSÉS)")
