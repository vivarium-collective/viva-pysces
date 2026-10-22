"""Smoke tests for the PySCeS process-bigraph wrappers.

These compile a tiny SBML model through PySCeS, so they require ``pysces`` +
``python-libsbml`` to be installed (the package's declared deps).
"""
import os
import textwrap

import pytest

pytest.importorskip("pysces")

from process_bigraph import allocate_core

from viva_pysces.processes import PyscesSteadyStateStep, PyscesUTCStep

# A minimal irreversible decay A -> B, k=1.0, A(0)=10, B(0)=0.
_SBML = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="decay">
        <listOfCompartments>
          <compartment id="c" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="c" initialConcentration="10"
                   hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
          <species id="B" compartment="c" initialConcentration="0"
                   hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k" value="1.0" constant="true"/>
        </listOfParameters>
        <listOfReactions>
          <reaction id="R" reversible="false">
            <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
            <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>
    """
)


@pytest.fixture()
def sbml_file(tmp_path):
    p = tmp_path / "decay.xml"
    p.write_text(_SBML, encoding="utf-8")
    return str(p)


def test_utc_step_shape_and_decay(sbml_file, tmp_path, monkeypatch):
    # Keep the conversion cache inside the test's tmp dir.
    monkeypatch.setenv("PBG_PYSCES_CACHE", str(tmp_path / "cache"))
    import importlib
    import viva_pysces.processes as proc
    importlib.reload(proc)

    step = proc.PyscesUTCStep({}, core=allocate_core())
    out = step.update({"model_source": sbml_file, "time": 5.0, "n_points": 20})
    res = out["result"]

    assert set(res.keys()) == {"time", "columns", "values"}
    assert len(res["time"]) == 20
    assert res["time"][0] == pytest.approx(0.0)
    assert res["time"][-1] == pytest.approx(5.0)
    assert {"A", "B"} <= set(res["columns"])
    # Every row has one value per column.
    assert all(len(row) == len(res["columns"]) for row in res["values"])

    ai = res["columns"].index("A")
    # A decays from 10 toward 0.
    assert res["values"][0][ai] == pytest.approx(10.0, rel=1e-3)
    assert res["values"][-1][ai] < res["values"][0][ai]


def test_steady_state_step(sbml_file, tmp_path, monkeypatch):
    monkeypatch.setenv("PBG_PYSCES_CACHE", str(tmp_path / "cache"))
    import importlib
    import viva_pysces.processes as proc
    importlib.reload(proc)

    step = proc.PyscesSteadyStateStep({}, core=allocate_core())
    out = step.update({"model_source": sbml_file})
    res = out["result"]
    assert res["kind"] == "steady_state"
    assert res["time"] is None
    assert "A" in res["observables"] and "B" in res["observables"]


# --------------------------------------------------------------------------- #
# Stiff-integration robustness: CVODE fallback + honest failure                #
# --------------------------------------------------------------------------- #

import numpy as np  # noqa: E402

from viva_pysces import processes as proc  # noqa: E402


class _DummyModel:
    """Minimal stand-in for a loaded PySCeS model.

    Returns ``lsoda_arr`` while ``mode_integrator`` is non-CVODE and
    ``cvode_arr`` once it is switched to CVODE, letting us drive the fallback
    logic deterministically without a real stiff integration.
    """

    def __init__(self, lsoda_arr, cvode_arr=None, labels=("Time", "A")):
        self.mode_integrator = "LSODA"
        self._lsoda = np.asarray(lsoda_arr, dtype=float)
        self._cvode = None if cvode_arr is None else np.asarray(cvode_arr, dtype=float)
        self._labels = list(labels)
        self.doSim_calls = []
        model = self

        class _DS:
            def getSpecies(self, lbls=True):
                use_cvode = str(model.mode_integrator).upper() == "CVODE"
                arr = model._cvode if use_cvode else model._lsoda
                if arr is None:  # CVODE requested but no finite data supplied
                    arr = model._lsoda
                return arr, model._labels

        self.data_sim = _DS()

    def doSim(self, end, points):
        self.doSim_calls.append((str(self.mode_integrator), end, points))


def test_species_all_nonfinite_detector():
    assert proc._species_all_nonfinite([[0.0, np.nan], [1.0, np.nan]]) is True
    assert proc._species_all_nonfinite([[0.0, np.inf], [1.0, -np.inf]]) is True
    assert proc._species_all_nonfinite([[0.0, 1.0], [1.0, np.nan]]) is False
    assert proc._species_all_nonfinite([[0.0, 10.0], [1.0, 5.0]]) is False
    assert proc._species_all_nonfinite([]) is True


def test_simulate_utc_raises_honest_failure_when_no_cvode(monkeypatch):
    """All-NaN LSODA trace + no CVODE must raise, not return a fake 'ok' trace."""
    monkeypatch.setattr(proc, "_assimulo_status",
                        lambda: (False, "No module named 'assimulo'"))
    model = _DummyModel(lsoda_arr=[[0.0, np.nan], [1.0, np.nan]])
    with pytest.raises(RuntimeError) as exc:
        proc._simulate_utc(model, end=10.0, points=2, quiet=True)
    msg = str(exc.value)
    assert "no finite trajectory" in msg
    assert "CVODE unavailable" in msg


def test_simulate_utc_retries_with_cvode(monkeypatch):
    """LSODA all-NaN but CVODE available -> switch and return the finite trace."""
    monkeypatch.setattr(proc, "_assimulo_status", lambda: (True, ""))
    model = _DummyModel(
        lsoda_arr=[[0.0, np.nan], [1.0, np.nan]],
        cvode_arr=[[0.0, 10.0], [1.0, 4.0]],
    )
    arr, labels, integrator = proc._simulate_utc(model, end=1.0, points=2, quiet=True)
    assert integrator == "CVODE"
    assert np.isfinite(arr[:, 1:]).all()
    # It tried LSODA first (fast path), then CVODE.
    assert [c[0] for c in model.doSim_calls] == ["LSODA", "CVODE"]


def test_simulate_utc_fast_path_no_retry(monkeypatch):
    """A finite first-pass trace returns immediately without touching CVODE."""
    called = {"n": 0}

    def _status():
        called["n"] += 1
        return (True, "")

    monkeypatch.setattr(proc, "_assimulo_status", _status)
    model = _DummyModel(lsoda_arr=[[0.0, 10.0], [1.0, 4.0]])
    arr, labels, integrator = proc._simulate_utc(model, end=1.0, points=2, quiet=True)
    assert integrator == "LSODA"
    assert called["n"] == 0  # CVODE availability never consulted
    assert len(model.doSim_calls) == 1


def test_utc_step_surfaces_failure(monkeypatch, tmp_path):
    """PyscesUTCStep.update must propagate the honest failure (no all-NaN leaf)."""
    monkeypatch.setattr(proc, "_load_model",
                        lambda src, quiet=True: _DummyModel([[0.0, np.nan]]))
    monkeypatch.setattr(proc, "_assimulo_status", lambda: (False, "no assimulo"))
    step = proc.PyscesUTCStep({}, core=allocate_core())
    with pytest.raises(RuntimeError):
        step.update({"model_source": "ignored", "time": 5.0, "n_points": 2})


@pytest.mark.skipif(
    not proc._assimulo_status()[0] or not os.environ.get("PYSCES_STIFF_SBML"),
    reason="needs assimulo/CVODE and PYSCES_STIFF_SBML pointing at a stiff SBML",
)
def test_cvode_rescues_real_stiff_model(tmp_path, monkeypatch):
    """Guarded: on a real stiff SBML, LSODA yields all-NaN but CVODE is finite."""
    import os as _os
    monkeypatch.setenv("PBG_PYSCES_CACHE", str(tmp_path / "cache"))
    sbml = _os.environ["PYSCES_STIFF_SBML"]
    model = proc._load_model(sbml, quiet=True)
    model.mode_integrator = "LSODA"  # start on the failing integrator
    arr, labels, integrator = proc._simulate_utc(model, end=100.0, points=200, quiet=True)
    assert integrator == "CVODE"
    assert np.isfinite(np.asarray(arr, dtype=float)[:, 1:]).any()
