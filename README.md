# viva-pysces

Process-bigraph wrapper for [PySCeS](https://github.com/PySCeS/pysces) (the
Python Simulator for Cellular Systems).

PySCeS reads its own `.psc` model format, so SBML is converted to PSC on first
use (via libSBML) and the result is cached on disk, content-addressed by SBML
source — repeated loads of the same model are fast. The cache lives in
`pysces_models/` under the working directory (override with `PBG_PYSCES_CACHE`),
mirroring AMICI's `amici_models/` convention.

## Components

| Class | Kind | Contract |
|-------|------|----------|
| `PyscesUTCStep` | one-shot Step | `inputs {model_source, time, n_points}` → `{result: {time, columns, values}}` |
| `PyscesSteadyStateStep` | one-shot Step | `inputs {model_source}` → `{result: {kind, time, observables}}` |
| `PyscesUTCProcess` | time-driven Process | convenience wrapper for the single-process composite |

The one-shot Steps speak the canonical multi-simulator comparison contract
(identical shape to `pbg-amici`'s `AmiciUTCStep` / `AmiciSteadyStateStep`), so
PySCeS drops straight into the `pbg-biomodels` batch-comparison harness.

## Install

```bash
pip install -e .            # pulls pysces + python-libsbml
pytest                      # smoke tests (compile a tiny decay model)
```

## Stiff models: CVODE fallback + honest failure

PySCeS' default **LSODA** integrator fails on stiff systems by returning an
**all-NaN** trajectory rather than raising. Previously the wrapper packaged that
into a `{result: …}` leaf and reported success, so downstream recorded an
all-NaN "ok" result.

The uniform-time-course path now:

1. runs LSODA first (non-stiff models keep their fast path untouched);
2. if the species trajectory comes back entirely non-finite, and **CVODE**
   (Assimulo/SUNDIALS) is importable, retries once with
   `mod.mode_integrator = 'CVODE'` — CVODE solves many stiff models LSODA cannot;
3. if no integrator yields a finite trajectory, **raises** `RuntimeError` so the
   caller records an honest failure instead of an all-NaN success.

CVODE needs Assimulo + SUNDIALS, which are most reliably installed via conda
(`conda install -c conda-forge assimulo`); a plain `pip install assimulo` must
build against a compatible SUNDIALS. When Assimulo is absent, stiff models fail
loudly (step 3) rather than silently.

The reusable helper `viva_pysces.processes._simulate_utc(model, end, points,
quiet)` (also exported from the package) carries this logic, so adapters that
drive PySCeS directly (e.g. `viva-biomodels`' `BiomodelsPyscesStep`) can get the
same robustness by integrating through it.

## Caveat: events

PySCeS ignores SBML *events* unless [Assimulo](https://jmodelica.org/assimulo)
is installed — it falls back to LSODA and silently drops events. Event-driven
models will therefore diverge from event-aware engines (COPASI, Tellurium,
AMICI). This matches the behaviour of the PySCeS reference results in the
BioSimulators test suite.
