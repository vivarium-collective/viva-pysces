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

## Caveat: events

PySCeS ignores SBML *events* unless [Assimulo](https://jmodelica.org/assimulo)
is installed — it falls back to LSODA and silently drops events. Event-driven
models will therefore diverge from event-aware engines (COPASI, Tellurium,
AMICI). This matches the behaviour of the PySCeS reference results in the
BioSimulators test suite.
