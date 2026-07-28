import csv, re
from theory.jones import braid_index
CHI={'3_1':-2,'4_1':0,'5_1':2,'5_2':3,'6_1':1,'6_2':-1,'6_3':0,'7_1':2,'7_2':0,
     '7_3':1,'7_4':-1,'7_5':1,'7_6':0,'8_1':0,'8_2':1,'8_3':0,'9_1':2,'7_7':None,'10_1':None}
print("="*60); print("AUDIT (C) — EXISTENCE / RÉALISABILITÉ"); print("="*60)
print("Test: (1) spin-statistique (fermion ⟺ J demi-entier ⟺ nb b-pair impair)")
print("      (2) charge Q=Σχ+Lk réalisable (Lk entier)\n")
rows=list(csv.reader(open("docs/catalogue_sigma_chi.csv")))
hdr=rows[0]
bad=[]
for r in rows[1:]:
    nm,cat,cfg,J,Q=r[0],r[1],r[2],r[3],r[4]
    if 'onde' in cfg or cfg=='---' or not re.search(r'\d+_\d+',cfg): continue
    ks=re.findall(r'\d+_\d+',cfg)
    J=float(J); Q=int(Q)
    # spin-statistique
    beven=sum(1 for k in ks if braid_index(k)%2==0)
    is_fermion_stat = (beven%2==1)
    is_fermion_spin = (abs(J*2)%2==1)  # J demi-entier
    stat_ok = (is_fermion_stat==is_fermion_spin)
    # charge : Σχ connu ?
    chis=[CHI.get(k) for k in ks]
    if None in chis:
        charge_ok='?(χ inconnu)'
    else:
        Sx=sum(chis); Lk=Q-Sx; charge_ok=(f'Lk={Lk} ✓' if isinstance(Lk,int) else 'NON')
    flag='' if stat_ok and 'NON' not in str(charge_ok) else '  ⚠️'
    if flag: bad.append(nm)
    print(f"  {nm:9s} {cfg:20s} J={J} Q={Q:+d} | stat:{'✓' if stat_ok else 'NON'} charge:{charge_ok}{flag}")
print(f"\n→ {len(bad)} configs problématiques" if bad else "\n→ TOUTES les configs réalisables (spin-stat + charge cohérents)")
if bad: print("  ",bad)
