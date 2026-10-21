"""Studio job history: persistence, restart recovery, pruning.

Run with the ComfyUI embedded interpreter:
python tests/test_studio_jobs.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
STUDIO_APP = PLUGIN_ROOT / "studio" / "server" / "app.py"


def _load_studio_app():
    """Import the Studio server module without starting the server or the engine."""
    spec = importlib.util.spec_from_file_location("studio_app_under_test", STUDIO_APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["studio_app_under_test"] = module
    spec.loader.exec_module(module)
    return module


APP = _load_studio_app()


def _job(module, job_id: str, status: str, created: float) -> dict:
    return {"id": job_id, "task": "file_download", "params": {}, "status": status,
            "error": None, "created": created, "started": created, "finished": created,
            "progress": None, "result": {}}


class JobHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="studio_jobs_"))
        self.jobs_file = self.root / "jobs.json"
        patcher = patch.multiple(APP, JOBS_FILE=self.jobs_file,
                                 JOBS={}, JOBS_MAX=4)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))

    ids = ()

    def _fill(self, *statuses) -> None:
        """Ids must look like production ids: 12 hex chars (validated on restore)."""
        self.ids = tuple(f"abc{index:09x}" for index in range(len(statuses)))
        for index, status in enumerate(statuses):
            job = _job(APP, self.ids[index], status, created=1000.0 + index)
            APP.JOBS[job["id"]] = job

    def test_round_trip_keeps_the_ui_payload_shape(self) -> None:
        self._fill("done", "error")
        APP._write_jobs()
        raw = json.loads(self.jobs_file.read_text("utf-8"))
        self.assertEqual(raw["version"], 1)
        self.assertEqual([j["id"] for j in raw["jobs"]], list(self.ids))
        self.assertEqual(set(raw["jobs"][0]), set(APP._JOB_FIELDS))

        APP.JOBS.clear()
        restored = APP._load_jobs()
        self.assertEqual(restored, 2)
        self.assertEqual(APP.JOBS[self.ids[1]]["status"], "error")

    def test_interrupted_jobs_become_visible_errors(self) -> None:
        self._fill("done", "running", "pending")
        APP._write_jobs()
        APP.JOBS.clear()
        APP._load_jobs()
        for job_id in self.ids[1:3]:
            job = APP.JOBS[job_id]
            self.assertEqual(job["status"], "error")
            self.assertIn("重启", job["error"])
        self.assertEqual(APP.JOBS[self.ids[0]]["status"], "done", "finished jobs keep their result")

    def test_prune_drops_oldest_terminal_records_only(self) -> None:
        self._fill("done", "done", "done", "done", "running")
        dropped = APP._prune_jobs()
        self.assertEqual(len(APP.JOBS), 4)
        self.assertEqual(dropped, [self.ids[0]])
        self.assertIn("running", [j["status"] for j in APP.JOBS.values()],
                      "an in-flight job is never dropped")

    def test_marking_dirty_does_not_write_synchronously(self) -> None:
        self._fill("done")
        APP._mark_dirty()
        self.assertFalse(self.jobs_file.exists(), "the writer thread owns the disk")
        self.assertTrue(APP._PERSIST["dirty"].is_set())

    def test_restored_records_are_validated(self) -> None:
        records = [
            {"id": "abc123abc123", "status": "done", "created": 1.0},
            {"id": "..\\..\\..\\Windows", "status": "done", "created": 2.0},   # path-shaped id
            {"id": "zzz", "status": "done", "created": 3.0},                    # wrong shape
        ]
        self.jobs_file.write_text(json.dumps({"version": 1, "jobs": records}), encoding="utf-8")
        self.assertEqual(APP._load_jobs(), 1)
        self.assertEqual(list(APP.JOBS), ["abc123abc123"])

    def test_restored_result_urls_must_stay_local(self) -> None:
        dirty = {"version": 1, "jobs": [{
            "id": "abc123abc123", "status": "done", "created": 1.0,
            "result": {"outputs": [
                {"name": "a.bin", "url": "/files/abc123abc123/a.bin"},
                {"name": "b.bin", "url": "javascript:alert(1)"},
            ]}}]}
        self.jobs_file.write_text(json.dumps(dirty), encoding="utf-8")
        APP._load_jobs()
        urls = [item["url"] for item in APP.JOBS["abc123abc123"]["result"]["outputs"]]
        self.assertEqual(urls, ["/files/abc123abc123/a.bin", ""])

    def test_file_lookup_is_contained_by_real_paths(self) -> None:
        out = self.root / "output"
        (out / "job1").mkdir(parents=True)
        (self.root / "output_evil").mkdir()
        (self.root / "output_evil" / "secret.html").write_text("x", encoding="utf-8")
        with patch.object(APP, "OUTPUT_DIR", out):
            self.assertEqual(APP._contained_output("job1/a.bin"), (out / "job1" / "a.bin").resolve())
            self.assertIsNone(APP._contained_output("../output_evil/secret.html"),
                              "a sibling directory with a matching prefix must not pass")

    def test_unreadable_history_starts_empty_instead_of_crashing(self) -> None:
        self.jobs_file.write_text("{not json", encoding="utf-8")
        self.assertEqual(APP._load_jobs(), 0)
        self.assertEqual(APP.JOBS, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
