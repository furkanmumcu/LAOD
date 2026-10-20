"""Which metrics are defined on which dataset.

SNAP and SemLoc-AP read ground-truth *names*. COCO-OOD has none -- every
instance carries the single category ``unknow object`` -- so both metrics are
undefined there while CAAP and U-AP remain perfectly well defined.

That rule used to live as seven ``== "coco_ood"`` conditionals scattered
through the report builders, i.e. downstream of the scorers, which is why
``maxdets_compare.py --snap --dataset coco_ood`` wrote a SNAP column anyway.
Under CLIP text embeddings that column read ~0.55 and looked like a result;
it was the cone effect, and CLIP's own label-shuffle control reproduced it to
four decimals. These tests pin the rule at its single source and check that
the scorers consult it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from laod.data.annotations import has_semantic_labels

ROOT = Path(__file__).resolve().parents[1]


class TestHasSemanticLabels:
    @pytest.mark.parametrize("dataset", ["coco", "lvis"])
    def test_named_datasets(self, dataset):
        assert has_semantic_labels(dataset) is True

    def test_coco_ood_is_unnamed(self):
        assert has_semantic_labels("coco_ood") is False

    def test_unknown_dataset_is_not_assumed_named(self):
        """A dataset nobody has classified must not default to 'has names'.

        Defaulting the other way would silently re-open exactly the hole this
        predicate closes: a new single-category benchmark would be scored on
        SNAP without anyone deciding that it should be.
        """
        assert has_semantic_labels("some_future_benchmark") is False


class TestScorersConsultThePolicy:
    """The guard has to sit in the scorers, not only in the reports.

    Checked statically so the test needs no GPU, no annotations and no cached
    runs -- the failure these guard against is a line being deleted, which a
    source check catches as well as an end-to-end run would.
    """

    #: Every entry point that both computes SNAP (or SemLoc-AP) and accepts a
    #: dataset argument. ``snap_sbert_all.py`` is deliberately absent: its
    #: COCO-OOD group is the shuffle control that establishes the exclusion,
    #: it takes no --dataset, and no report displays it.
    SCORERS = ["scripts/maxdets_compare.py", "scripts/semloc_all.py",
               "scripts/report_stage.py", "scripts/rescore_vlm.py",
               "scripts/run_vlm_detect.py", "run_benchmark.py",
               "eval_metrics_only.py"]

    @pytest.mark.parametrize("script", SCORERS)
    def test_scorer_imports_and_calls_the_predicate(self, script):
        src = (ROOT / script).read_text()
        tree = ast.parse(src)
        imported = any(
            isinstance(n, ast.ImportFrom)
            and n.module == "laod.data.annotations"
            and any(a.name == "has_semantic_labels" for a in n.names)
            for n in ast.walk(tree))
        called = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "has_semantic_labels"
            for n in ast.walk(tree))
        assert imported, f"{script} does not import has_semantic_labels"
        assert called, f"{script} imports the predicate but never calls it"

    @pytest.mark.parametrize(
        "script", ["build_final_report.py", "build_intervals_report.py"])
    def test_reports_carry_no_hardcoded_dataset_check(self, script):
        """No literal comparison against the dataset name should survive.

        The point of the predicate is that there is one place to change when
        another single-category benchmark is added.
        """
        src = (ROOT / "scripts" / script).read_text()
        for bad in ('== "coco_ood"', '!= "coco_ood"'):
            assert bad not in src, f"{script} still hardcodes `{bad}`"


class TestShippedArtefacts:
    """No committed COCO-OOD table may carry a SNAP column.

    This is the check that was missing. ``main_coco_ood.csv`` was the obvious
    offender, but ``phase3_vlm_coco_ood.csv`` carried one too -- written by a
    different script, read by no report, and so invisible until someone went
    looking. Globbing rather than naming files is the point: a new COCO-OOD
    table is covered the day it is added.
    """

    def test_no_coco_ood_table_has_a_snap_column(self):
        import csv
        tables = sorted((ROOT / "results").glob("*coco_ood*.csv"))
        assert tables, "no COCO-OOD tables found; has results/ moved?"
        offenders = {}
        for t in tables:
            with t.open() as fh:
                cols = next(csv.reader(fh))
            if snap := [c for c in cols if c.startswith("snap")]:
                offenders[t.name] = snap
        assert not offenders, offenders

    def test_the_shuffle_control_is_still_null(self):
        """snap_sbert.csv keeps its COCO-OOD rows on purpose.

        They are the evidence the exclusion is correct, not a result: a
        shuffled-label control returning a gain of exactly zero. Guard the
        reason they are allowed to stay -- if they ever became a nonzero
        number, they would be a result and would have to go.
        """
        import csv
        p = ROOT / "results" / "snap_sbert.csv"
        if not p.exists():
            pytest.skip("snap_sbert.csv not present")
        s4 = [r for r in csv.DictReader(p.open()) if r["stage"] == "stage4"]
        assert s4, "the COCO-OOD control rows have gone missing"
        assert all(abs(float(r["snap_macro"])) < 1e-6 for r in s4)
        assert all(abs(float(r["gain_macro"])) < 1e-6 for r in s4)


class TestSnapIsRefusedOnCocoOod:
    def test_cli_declines_and_writes_no_snap_columns(self, tmp_path):
        """End-to-end on the CLI, with no runs to score.

        ``--snap`` must be reported as ignored before anything expensive
        happens: the notice has to come from the policy check, not from a
        downstream failure to find an encoder or a checkpoint.
        """
        if not (ROOT / "datasets" / "annotations").exists():
            pytest.skip("annotations not provisioned")
        out = tmp_path / "ood.csv"
        r = subprocess.run(
            [sys.executable, "scripts/maxdets_compare.py",
             "--dataset", "coco_ood", "--snap",
             "--pattern", str(tmp_path / "no-such-run-*"),
             "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert "--snap ignored" in r.stdout, r.stdout + r.stderr
        if out.exists():
            assert "snap" not in out.read_text().splitlines()[0]
