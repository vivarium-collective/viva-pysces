"""PyscesFullReportUTCStep: one-shot full-report UTC on the engine-core contract."""
import textwrap

import pytest

pytest.importorskip("pysces")

from engine_core import UniformTimeCourseStep
from process_bigraph import allocate_core

from viva_pysces.processes import PyscesFullReportUTCStep

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


@pytest.fixture
def sbml_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PBG_PYSCES_CACHE", str(tmp_path / "cache"))
    p = tmp_path / "decay.xml"
    p.write_text(_SBML, encoding="utf-8")
    return str(p)


def _step():
    core = allocate_core()
    core.register_link("PyscesFullReportUTCStep", PyscesFullReportUTCStep)
    return PyscesFullReportUTCStep({}, core=core)


def test_subclasses_the_engine_core_contract():
    assert issubclass(PyscesFullReportUTCStep, UniformTimeCourseStep)
    step = _step()
    assert step.inputs() == {
        "model_source": "string",
        "time": "float",
        "n_points": "integer",
    }
    assert step.outputs() == {"result": "tree"}


def test_emits_canonical_full_report_shape(sbml_file):
    out = _step().update({"model_source": sbml_file, "time": 5.0, "n_points": 11})
    r = out["result"]
    assert set(r) == {"time", "columns", "values"}
    assert len(r["time"]) == 11
    assert len(r["values"]) == 11
    assert all(len(row) == len(r["columns"]) for row in r["values"])
    assert all(isinstance(v, float) for row in r["values"] for v in row)


def test_reports_species_fluxes_and_compartments(sbml_file):
    """The full report carries species, reaction fluxes AND compartments."""
    r = _step().update({"model_source": sbml_file, "time": 5.0, "n_points": 6})["result"]
    cols = set(r["columns"])
    assert {"A", "B"} <= cols   # species
    assert "R" in cols          # reaction flux
    assert "c" in cols          # compartment


def test_rejects_degenerate_n_points(sbml_file):
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        _step().update({"model_source": sbml_file, "time": 5.0, "n_points": 1})
