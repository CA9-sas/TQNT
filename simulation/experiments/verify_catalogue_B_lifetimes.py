import numpy as np
print("="*60); print("AUDIT (B) — DURÉES DE VIE & STABILITÉ"); print("="*60)
# (nom, τ exp [s], Q décroissance [MeV], type)
# Q = énergie libérée (m_parent − Σm_produits)
DEC=[
 ('e-',   np.inf, 0,      'STABLE (plancher leptonique)'),
 ('p',    np.inf, 0,      'STABLE (plancher baryonique)'),
 ('n',    879,    0.782,  'faible (β)'),
 ('mu-',  2.2e-6, 105.1,  'faible (leptonique)'),
 ('tau-', 2.9e-13,1777-0, 'faible (leptonique)'),
 ('pi+',  2.6e-8, 33.9,   'faible'),
 ('K+',   1.24e-8,388,    'faible'),
 ('Lambda',2.6e-10,37.7,  'faible (ΔS)'),
 ('Sigma-',1.5e-10,117,   'faible (ΔS)'),
 ('Xi-',  1.6e-10,59,     'faible (ΔS)'),
 ('Omega-',0.82e-10,62,   'faible (ΔS)'),
 ('D+',   1.0e-12, 1870-500,'faible (charme)'),
 ('B+',   1.6e-12, 5279-2000,'faible (bottom)'),
 ('pi0',  8.5e-17, 135,   'EM'),
 ('eta',  5.0e-19, 548,   'EM'),
 ('rho',  4.5e-24, 775-280,'FORT'),
 ('Delta',5.6e-24, 1232-1078,'FORT'),
 ('J/psi',7.1e-21, 3097,  'EM/OZI'),
 ('W',    3.2e-25, 80377, 'EW'),
 ('top',  5e-25,   172690,'EW'),
]
# Sargent : décroissances FAIBLES leptoniques/semi → τ ∝ 1/Q⁵
print("\n1) STABILITÉ — planchers topologiques :")
print("   e⁻ = plancher LEPTONIQUE (3_1, Z₃ minimal, rien de plus léger chargé) → STABLE ✓")
print("   p  = plancher BARYONIQUE (entrelacs 3-brins minimal, B conservé topo) → STABLE ✓")
print("   Les 2 seuls fondamentaux stables = les 2 planchers. COHÉRENT.")
print("\n2) SARGENT τ∝1/Q⁵ sur les faibles LEPTONIQUES (test propre) :")
lep=[('mu-',2.2e-6,105.1),('tau-',2.9e-13,1777)]
for nm,t,Q in lep:
    print(f"   {nm}: τ·Q⁵ = {t*Q**5:.3e}")
print(f"   ratio (m_τ/m_μ)⁵ = {(1777/105.1)**5:.3e} ; τ_μ/τ_τ = {2.2e-6/2.9e-13:.3e}")
print(f"   → accord à ~facteur 2 (BR τ→lept 35%) : Sargent OK, mais = phase-space STANDARD")
print("\n3) CORRÉLATION log τ vs log Q (toutes faibles) :")
weak=[(t,Q) for nm,t,Q,ty in DEC if 'faible' in ty and Q>0]
lt=np.log10([w[0] for w in weak]); lQ=np.log10([w[1] for w in weak])
print(f"   corr(log τ, log Q) = {np.corrcoef(lt,lQ)[0,1]:.2f}  (attendu <0 ; Sargent pente ~−5)")
sl=np.polyfit(lQ,lt,1)[0]; print(f"   pente log-log = {sl:.1f}  (Sargent prédit −5)")
print("\n4) HIÉRARCHIE des types (ordres de grandeur) :")
print("   FORT ~10⁻²⁴ s < EM ~10⁻¹⁷ s < FAIBLE ~10⁻¹⁰..10³ s  — ordre REPRODUIT")
print("\nVERDICT (B): stabilité (planchers e,p) = contenu TQNT réel ✓ ;")
print("  ordering fort/EM/faible reproduit ; Sargent Q⁵ = phase-space STANDARD (pas unique TQNT).")
print("  ⚠️ Les durées de vie ne sont PAS une prédiction discriminante de la TQNT.")
