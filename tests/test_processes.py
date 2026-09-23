"""Smoke tests for the PySCeS process-bigraph wrappers.

These compile a tiny SBML model through PySCeS, so they require ``pysces`` +
``python-libsbml`` to be installed (the package's declared deps).
"""
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
