"""PySCeS Process / Step wrappers for process-bigraph.

Wraps `PySCeS <https://github.com/PySCeS/pysces>`_ (the Python Simulator for
Cellular Systems) as process-bigraph Steps. PySCeS reads its own ``.psc`` model
format, so SBML is converted to PSC on first use (via libSBML) and the compiled
model is cached on disk, content-addressed by SBML source — repeated loads of
the same model are fast.

Two one-shot Steps share the load path and speak the canonical multi-simulator
comparison contract:

* :class:`PyscesUTCStep` — uniform time course →
  ``{result: {time, columns, values}}``.
* :class:`PyscesSteadyStateStep` — steady-state solve →
  ``{result: {kind: "steady_state", time: None, observables: {id: value}}}``.

:class:`PyscesUTCProcess` is a thin time-driven convenience Process for the
single-process composite; the comparison harness uses the Steps.

Stiff models: PySCeS' default LSODA integrator fails on stiff systems by
returning an all-NaN trajectory (not by raising). The uniform-time-course path
detects such an all-non-finite result and, when CVODE (Assimulo/SUNDIALS) is
available, retries once with ``mode_integrator = 'CVODE'``. Non-stiff models
keep the fast LSODA path untouched. If no integrator yields a finite trajectory
the Step raises instead of emitting an all-NaN "success", so callers record an
honest failure.

Caveat: PySCeS ignores SBML *events* unless Assimulo is installed (it falls back
to LSODA). Event-driven models will therefore differ from event-aware engines.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from typing import Any

from process_bigraph import Process, Step

#: Persistent cache for converted PSC models, content-addressed by SBML source.
#: Override with ``PBG_PYSCES_CACHE``; defaults to ``pysces_models/`` under cwd
#: (mirrors AMICI's ``amici_models/`` convention so re-runs reuse conversions).
_CACHE_ROOT = Path(os.environ.get("PBG_PYSCES_CACHE", "pysces_models"))


def _hash_source(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


@contextlib.contextmanager
def _maybe_quiet(enabled: bool):
    """Silence PySCeS' verbose parser/solver chatter on stdout/stderr."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _validate_n_points(n: Any) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise ValueError(f"PyscesUTCStep: n_points must be an integer >= 2, got {n!r}")
    if n < 2:
        raise ValueError(f"PyscesUTCStep: n_points must be >= 2, got {n}")
    return n


def _assimulo_status() -> tuple[bool, str]:
    """Return ``(have_cvode, detail)`` for PySCeS' optional CVODE backend.

    CVODE (a stiff BDF solver) is only available when ``assimulo``/SUNDIALS is
    importable; PySCeS records this at import time on ``PyscesModel``. When it is
    missing, ``detail`` carries PySCeS' own load error so callers can report it
    honestly instead of silently falling back to LSODA.
    """
    try:
        import pysces.PyscesModel as _pm

        have = bool(getattr(_pm, "_HAVE_ASSIMULO", False))
        detail = str(getattr(_pm, "_ASSIMULO_LOAD_ERROR", "") or "")
        return have, detail
    except Exception as exc:  # pragma: no cover - pysces import issues
        return False, f"{type(exc).__name__}: {exc}"


def _species_all_nonfinite(arr: Any) -> bool:
    """True when the species block of a PySCeS sim array has no finite value.

    PySCeS' default LSODA integrator fails on stiff models by returning an
    all-NaN (occasionally all-inf) trajectory rather than raising — the array
    is shaped ``[time, species...]`` with the first column the time points. A
    trajectory with no finite species value anywhere is unusable, which is the
    signal to retry with CVODE and, failing that, to report failure.
    """
    import numpy as np

    a = np.asarray(arr, dtype=float)
    if a.size == 0 or a.ndim != 2 or a.shape[1] < 2:
        return True
    return not bool(np.isfinite(a[:, 1:]).any())


def _simulate_utc(model, end: float, points: int, quiet: bool):
    """Integrate a uniform time course, retrying stiff failures with CVODE.

    Runs the model's default integrator (LSODA) first so non-stiff models keep
    their fast path unchanged. If that produces an all-non-finite trajectory (or
    raises), and CVODE is available, switches ``mode_integrator`` to CVODE and
    retries once. Returns ``(arr, labels, integrator_used)``.

    Raises :class:`RuntimeError` if no integrator produces a finite trajectory,
    so callers surface an honest failure instead of an all-NaN "success".
    """
    def _run():
        with _maybe_quiet(quiet):
            model.doSim(end=end, points=points)
            arr, labels = model.data_sim.getSpecies(lbls=True)
        return arr, labels

    start_integrator = str(getattr(model, "mode_integrator", "LSODA") or "LSODA")
    last_err: Exception | None = None
    try:
        arr, labels = _run()
        if not _species_all_nonfinite(arr):
            return arr, labels, start_integrator
        last_err = RuntimeError(
            f"{start_integrator} produced an all-non-finite trajectory"
        )
    except Exception as exc:
        last_err = exc

    have_cvode, detail = _assimulo_status()
    if have_cvode and start_integrator.upper() != "CVODE":
        try:
            model.mode_integrator = "CVODE"
            arr, labels = _run()
            if not _species_all_nonfinite(arr):
                return arr, labels, "CVODE"
            last_err = RuntimeError("CVODE produced an all-non-finite trajectory")
        except Exception as exc:
            last_err = exc

    hint = (
        "CVODE unavailable (assimulo/SUNDIALS not importable: "
        f"{detail or 'unknown'})"
        if not have_cvode
        else "CVODE also failed"
    )
    raise RuntimeError(
        f"PySCeS produced no finite trajectory; {hint}. Last error: {last_err!r}"
    )


def _load_model(sbml_path: str, quiet: bool = True):
    """Convert (if needed) an SBML file to PSC and return a loaded pysces model.

    The conversion + load is cached under ``_CACHE_ROOT/<sha>`` keyed by the
    SBML source text, so repeated calls for the same model skip reconversion.
    """
    import pysces

    sbml_path = os.path.abspath(sbml_path)
    if not os.path.isfile(sbml_path):
        raise ValueError(f"PySCeS: model_source is not a file: {sbml_path}")
    # SBML frequently carries non-ASCII metadata; read as UTF-8.
    source = Path(sbml_path).read_text(encoding="utf-8", errors="replace")
    key = _hash_source(source)
    cache = _CACHE_ROOT / key
    cache.mkdir(parents=True, exist_ok=True)
    psc_name = f"model_{key}.psc"
    psc_path = cache / psc_name

    with _maybe_quiet(quiet):
        if not psc_path.is_file():
            # PySCeS reads files with the locale default encoding (ascii under a
            # C locale), so SBML carrying non-ASCII metadata (µ, °, accented
            # names) raises UnicodeDecodeError. Convert from an ASCII-safe copy
            # — non-ASCII becomes XML numeric character references, which libSBML
            # parses identically — so loading is locale-independent.
            safe_name = f"src_{key}.xml"
            ascii_xml = source.encode("ascii", "xmlcharrefreplace").decode("ascii")
            (cache / safe_name).write_text(ascii_xml, encoding="ascii")
            # convertSBML2PSC writes ``<safe_name>.psc`` into pscdir.
            pysces.interface.convertSBML2PSC(
                safe_name, sbmldir=str(cache), pscdir=str(cache)
            )
            produced = cache / (safe_name + ".psc")
            if not produced.is_file():
                raise RuntimeError(
                    f"PySCeS: SBML→PSC conversion produced no file for {sbml_path}"
                )
            produced.replace(psc_path)
        model = pysces.model(psc_name, dir=str(cache))
    return model


# --------------------------------------------------------------------------- #
# One-shot Steps                                                              #
# --------------------------------------------------------------------------- #


class PyscesUTCStep(Step):
    """One-shot uniform time course via PySCeS.

    Runtime inputs (so a loader can feed them dynamically):

    * ``model_source`` — path to an SBML file.
    * ``time``         — integration horizon (t runs 0 → time).
    * ``n_points``     — number of evenly-spaced output points (>= 2).

    Emits the canonical ``{result: {time, columns, values}}`` trajectory, one
    column per floating-species id. Matches the one-shot UTC contract consumed
    by multi-simulator comparison harnesses.
    """

    config_schema = {"quiet": {"_type": "boolean", "_default": True}}

    def inputs(self) -> dict[str, str]:
        return {"model_source": "string", "time": "float", "n_points": "integer"}

    def outputs(self) -> dict[str, str]:
        return {"result": "tree"}

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        quiet = bool(self.config.get("quiet", True))
        n_points = _validate_n_points(state["n_points"])
        horizon = float(state["time"])

        model = _load_model(state["model_source"], quiet=quiet)
        # LSODA first (fast path); retry stiff all-NaN failures with CVODE and,
        # if nothing yields finite dynamics, raise so the result is recorded as
        # a failure rather than an all-NaN "ok" leaf.
        arr, labels, _integrator = _simulate_utc(model, horizon, n_points, quiet)

        # labels[0] == "Time"; remaining are species ids.
        columns = [str(c) for c in labels[1:]]
        times = [float(row[0]) for row in arr]
        values = [[float(row[i + 1]) for i in range(len(columns))] for row in arr]
        return {"result": {"time": times, "columns": columns, "values": values}}


class PyscesSteadyStateStep(Step):
    """One-shot steady-state solve via PySCeS.

    Runtime input ``model_source`` (an SBML file path). Solves for the steady
    state and emits each floating species' steady value as
    ``{result: {kind: "steady_state", time: None, observables: {id: value}}}``.
    """

    config_schema = {"quiet": {"_type": "boolean", "_default": True}}

    def inputs(self) -> dict[str, str]:
        return {"model_source": "string"}

    def outputs(self) -> dict[str, str]:
        return {"result": "tree"}

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        quiet = bool(self.config.get("quiet", True))
        model = _load_model(state["model_source"], quiet=quiet)
        with _maybe_quiet(quiet):
            model.doState()
            species_ids = [str(s) for s in model.species]
            ss_values = list(model.state_species)

        observables = {
            sid: float(val) for sid, val in zip(species_ids, ss_values)
        }
        return {
            "result": {
                "kind": "steady_state",
                "time": None,
                "observables": observables,
            }
        }


# --------------------------------------------------------------------------- #
# Time-driven convenience Process                                            #
# --------------------------------------------------------------------------- #


class PyscesUTCProcess(Process):
    """Time-driven PySCeS Process for the single-process composite.

    Lazily loads the model and, on each ``update``, integrates from t=0 to the
    accumulated time and reports the species concentrations at the horizon.
    Deterministic-model convenience wrapper — for tight time-coupling prefer a
    natively steppable engine; the comparison harness uses :class:`PyscesUTCStep`.

    Config:
        model_source: path to an SBML file.
        model_format: 'sbml' (only SBML is supported via conversion).
        quiet: suppress PySCeS stdout chatter (default True).
    """

    config_schema = {
        "model_source": {"_type": "string", "_default": ""},
        "model_format": {"_type": "string", "_default": "sbml"},
        "quiet": {"_type": "boolean", "_default": True},
    }

    def __init__(self, config: dict | None = None, core: Any = None):
        super().__init__(config=config, core=core)
        self._model = None
        self._species_ids = None
        self._elapsed = 0.0

    def _build(self):
        if self._model is not None:
            return
        src = self.config.get("model_source") or ""
        if not src:
            raise ValueError("PyscesUTCProcess requires config['model_source'].")
        self._model = _load_model(src, quiet=bool(self.config.get("quiet", True)))
        self._species_ids = [str(s) for s in self._model.species]

    def inputs(self) -> dict[str, str]:
        return {"species": "maybe[map[float]]"}

    def outputs(self) -> dict[str, str]:
        return {
            "species_concentrations": "overwrite[map[float]]",
            "time": "overwrite[float]",
        }

    def initial_state(self) -> dict[str, Any]:
        self._build()
        init = {
            sid: float(val)
            for sid, val in zip(self._species_ids, self._model.species_init)
        }
        return {"species_concentrations": init, "time": 0.0}

    def update(self, state: dict[str, Any], interval: float) -> dict[str, Any]:
        self._build()
        self._elapsed += float(interval)
        quiet = bool(self.config.get("quiet", True))
        arr, labels, _integrator = _simulate_utc(
            self._model, self._elapsed, 2, quiet
        )
        cols = [str(c) for c in labels[1:]]
        last = arr[-1]
        species = {cols[i]: float(last[i + 1]) for i in range(len(cols))}
        return {"species_concentrations": species, "time": float(self._elapsed)}
