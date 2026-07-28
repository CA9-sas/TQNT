# TQNT — Théorie Quantique des Nœuds Topologiques (V6)

**Topological Knot Quantum Theory** — a framework in which spacetime is the
undulating weave of a single Chern–Simons-type field χ, and elementary
particles are topological knots of its undulation. Mass, spin, charge and
lifetimes are read off knot invariants; gravitation emerges by depletion
(G = ħc/M_Pl², single equation); cosmology is mechanized (Ω_tot = 1 and
w = −1 as theorems).

**Author:** Lilian Alexandre Cariou — CA9 SAS, Melun, France
(lilian.cariou@ca9.fr, ORCID [0009-0006-8276-4048](https://orcid.org/0009-0006-8276-4048))

## Key results (V6, July 2026)

| Result | Value | Status |
|---|---|---|
| Mass formula `m = √σ_χ · 𝒮` | 89 masses, **0.13 % median** (226-state extension: 1.43 %) | 10 global parameters, 0 per particle |
| Koide relation | Q = 2/3 at 9.2 ppm; m_τ at +61 ppm; **m_e computed to ±3 %** (no parameters) | published ([DOI 10.5281/zenodo.20817001](https://doi.org/10.5281/zenodo.20817001)), submitted to EPJ Plus |
| Gravitation | G = ħc/M_Pl² (−0.05 % vs CODATA) | single equation; nucleonic formula = reading of M_Pl |
| Weak sector | Fermi structure derived (P15, g_topo = 1); VEV derived (+28 ppm) | no weak coupling constant |
| Desert theorem | 617/617 links of [12–80] GeV unstable (~1e-18 s) | the observed spectral emptiness is predicted |
| Ω_cc | observed by LHCb (~3727 MeV) **within the predicted range** [3715, 3788] | dated preregistration now archived |

## References (source of truth)

- **V6 monograph** (FR reorganized edition = reference; FR/EN editions included
  in `docs/`): [DOI 10.5281/zenodo.19131538](https://doi.org/10.5281/zenodo.19131538)
- **Complete particle catalogue**: `docs/TQNT_catalog_complete.xlsx` (15 sheets,
  incl. the 1539 links with V6 masses & lifetimes)
- **Preregistration of dated predictions** (FR/EN PDFs in `docs/`)
- **Koide paper v2** (FR/EN PDFs in `docs/`)
- Online presentation & free course (FR/EN): https://www.ca9.fr/recherche/

## Repository layout

```
theory/       canonical modules: Jones/A-polynomials, Chern–Simons,
              knot catalogue, ropelengths (ACPR), P15 reconnection,
              emergent metric, postulates & constants
simulation/   FEM Schrödinger solver + ~400 experiment scripts
              (each script = one derivation/verification, self-documented
              in French; filenames map to monograph sections)
tests/        pytest suite (~1900 tests)
docs/         publication PDFs, catalogue (xlsx + csv), figures
output/       CSV results consumed/produced by the scripts
```

## Install & run

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v                       # test suite
python3 -m simulation.experiments.<script_name>   # any experiment
```

Examples:

```bash
# canonical σ_χ spectrum engine (89 states, 0.13 % median)
python3 -m simulation.experiments.spectrum_sigmachi_full
# desert theorem taxonomy scan
python3 -m simulation.experiments.scan_taxonomie_entrelacs
# free-knot predictions (masses & lifetimes)
python3 -m simulation.experiments.predictions_noeuds_libres
# complete catalogue generator (xlsx)
python3 -m simulation.experiments.genere_catalogue_complet
```

## Epistemic discipline

Every claim carries a status — derived / theorem / conditional (derivation
under a declared identification) / identified / calibrated / order of
magnitude — and the programme keeps a public **register of its own
refutations** (11 dated entries). Assignment-based agreements must pass a
null test before being presented as evidence. See the monograph's
falsifiability chapter and the preregistration.

## License

- **Code** (`theory/`, `simulation/`, `tests/`): MIT
- **Documents & figures** (`docs/`): CC-BY 4.0

See `LICENSE` and `docs/LICENSE-CC-BY-4.0`.
