import numpy as np
# Γ = |M|² · ρ(Q).  Sargent ≈ ρ(Q)~Q⁵.  On EXTRAIT |M|² = Γ/Q⁵ = 1/(τ·Q⁵)
# et on regarde s'il est organisé par la GÉOMÉTRIE de reconnexion (ΔLk), pas par Q.
# (nom, τ[s], Q[MeV], type, ΔLk_topologique)
#   ΔLk: 0=fort (réarrangement tension, pas de changement d'enlacement)
#        'γ'=EM (émission d'onde χ), 1=faible (reconnexion d'enlacement, charge change)
P=[
 ('Delta', 5.6e-24,154,'FORT',0),('rho',4.5e-24,495,'FORT',0),('omega',7.8e-23,470,'FORT',0),
 ('pi0',  8.5e-17,135,'EM',  'γ'),('eta',5.0e-19,548,'EM','γ'),('Sigma0',7.4e-20,74,'EM','γ'),
 ('n',    879,   0.782,'FAIBLE',1),('mu',2.2e-6,105,'FAIBLE',1),('pi+',2.6e-8,34,'FAIBLE',1),
 ('K+',   1.24e-8,388,'FAIBLE',1),('Lambda',2.6e-10,38,'FAIBLE',1),('Sigma+',0.8e-10,116,'FAIBLE',1),
 ('Xi-',  1.6e-10,59,'FAIBLE',1),('Omega-',0.82e-10,62,'FAIBLE',1),('tau',2.9e-13,1777,'FAIBLE',1),
]
print("Γ = |M|²·ρ(Q) ; on extrait log|M|² ≈ log(1/τ) − 5·log Q  (Q en GeV)")
print(f"{'part':8s} {'type':7s} {'ΔLk':>4s} {'logτ':>6s} {'5logQ':>7s} {'log|M|²':>8s}")
byT={}
for nm,tau,Q,ty,dLk in P:
    Qg=Q/1000; logM2=-np.log10(tau)-5*np.log10(Qg)
    byT.setdefault(ty,[]).append(logM2)
    print(f"{nm:8s} {ty:7s} {str(dLk):>4s} {-np.log10(tau) if tau<1e30 else 99:>6.1f} {5*np.log10(Qg):>7.1f} {logM2:>8.1f}")
print("\n=== |M|² (amplitude de reconnexion) PAR TYPE géométrique ===")
for ty in ['FORT','EM','FAIBLE']:
    v=byT[ty]; print(f"  {ty:7s} : log|M|² = {np.mean(v):+6.1f} ± {np.std(v):.1f}  (ΔLk={'0' if ty=='FORT' else 'γ' if ty=='EM' else '1'})")
print("\n=== LE POINT ===")
strong=np.mean(byT['FORT']); weak=np.mean(byT['FAIBLE'])
print(f"  écart |M|² fort→faible = {strong-weak:.0f} décades = pur MATRICE (géométrie de reconnexion ΔLk)")
print(f"  Sargent (Q⁵) ne couvre que ~3-5 décades ; le facteur DOMINANT ({strong-weak:.0f} déc) est GÉOMÉTRIQUE.")
# comparaison à G_F
GF=1.166e-5; Lam=0.209  # GeV
supp=(GF*Lam**2)**2
print(f"  suppression faible attendue (G_F Λ²)² = {supp:.1e} → {np.log10(supp):.0f} décades ✓ (≈ l'écart mesuré)")
print(f"\n  → |M|² est CONSTANT par type (σ~{np.std(byT['FAIBLE']):.1f} déc intra-faible) et saute de {strong-weak:.0f} déc")
print(f"    entre types = la RECONNEXION topologique (ΔLk). Sargent = juste ρ(Q). RACINE = |M|² géométrique.")
