"""Review persistence, original-data protection and export readiness."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.offline_media import atomic_json
from src.review_project import ProjectStore, review_progress, browser_project
from src.review_server import ReviewServer


def project_fixture():
    return {"version": 1, "revision": 0, "name": "Recorrido · Trinidad", "updated_at": "now",
            "source": {"sha256": "source", "name": "clip.MOV"},
            "analysis": {"complete": True, "preview": "preview.mp4"}, "center": [-33.5, -56.9],
            "route": [], "events": [{"event_id": 1, "status": "pending", "position": None,
                                      "observation": {"event_id": 1, "confidence": .9, "timestamp": 1.2}}]}


class ReviewProjectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "project.json"
        atomic_json(self.path, project_fixture())
        self.store = ProjectStore(self.path)

    def test_save_reopen_preserves_turns_repeated_streets_and_manual_position(self):
        project = self.store.snapshot()
        project["route"] = [[-33.5, -56.9], [-33.501, -56.9], [-33.501, -56.901], [-33.5, -56.9]]
        project["events"][0].update(status="accepted", position=[-33.501, -56.9005])
        saved = self.store.save(project)
        reopened = ProjectStore(self.path).snapshot()
        self.assertEqual(reopened, saved)
        self.assertEqual(reopened["revision"], 1)
        self.assertEqual(reopened["route"], project["route"])
        self.assertTrue(review_progress(reopened)["ready"])
        self.assertEqual(reopened["events"][0]["observation"]["confidence"], .9)

    def test_export_requires_route_and_complete_review(self):
        with self.assertRaises(ValueError):
            self.store.export_snapshot()
        project = self.store.snapshot()
        project["route"] = [[-33.5, -56.9], [-33.501, -56.9]]
        self.store.save(project)
        with self.assertRaises(ValueError):
            self.store.export_snapshot()
        project = self.store.snapshot()
        project["events"][0]["status"] = "rejected"
        self.store.save(project)
        self.assertEqual(review_progress(self.store.export_snapshot())["accepted"], 0)

    def test_zero_events_still_requires_two_distinct_route_points(self):
        project = project_fixture()
        project["events"] = []
        project["route"] = [[-33.5, -56.9], [-33.5, -56.9]]
        self.assertFalse(review_progress(project)["ready"])
        project["route"].append([-33.501, -56.9])
        self.assertTrue(review_progress(project)["ready"])

    def test_invalid_coordinates_states_and_original_mutations_are_rejected(self):
        mutations = [lambda p: p["route"].append([float("nan"), 0]),
                     lambda p: p["route"].append([True, 0]),
                     lambda p: p["route"].append([91, 0]),
                     lambda p: p["events"][0].update(status="accepted"),
                     lambda p: p["events"][0].update(position=[-33.5, -56.9]),
                     lambda p: p["events"][0]["observation"].update(confidence=1),
                     lambda p: p["events"].clear(), lambda p: p["source"].update(sha256="other")]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                project = self.store.snapshot()
                mutate(project)
                with self.assertRaises(ValueError):
                    self.store.save(project)
                self.assertEqual(self.store.snapshot()["revision"], 0)

    def test_stale_tab_cannot_overwrite_newer_review(self):
        old = self.store.snapshot()
        self.store.save(self.store.snapshot())
        with self.assertRaises(RuntimeError):
            self.store.save(old)
        self.assertEqual(ProjectStore(self.path).snapshot()["revision"], 1)

    def test_browser_preserves_nanosecond_timestamp_without_modifying_original(self):
        project = project_fixture()
        timestamp = 1791237555658403500
        project["source"]["mtime_ns"] = timestamp
        atomic_json(self.path, project)
        store = ProjectStore(self.path)
        payload = browser_project(store.snapshot())
        self.assertEqual(payload["source"]["mtime_ns"], str(timestamp))
        payload["route"] = [[-33.5, -56.9], [-33.501, -56.9]]
        saved = store.save(payload)
        self.assertEqual(saved["source"]["mtime_ns"], timestamp)
        self.assertEqual(ProjectStore(self.path).snapshot()["source"]["mtime_ns"], timestamp)

    def test_existing_browser_draft_with_rounded_timestamp_can_save_route_and_review(self):
        project = project_fixture()
        project["source"]["mtime_ns"] = 1791237555658403500
        atomic_json(self.path, project)
        store = ProjectStore(self.path)
        payload = store.snapshot()
        payload["source"]["mtime_ns"] = 1791237555658403600  # actual JSON.stringify result
        payload["route"] = [[-33.5, -56.9], [-33.501, -56.9], [-33.501, -56.901]]
        payload["events"][0].update(status="accepted", position=[-33.501, -56.9005])
        saved = store.save(payload)
        self.assertEqual(saved["source"]["mtime_ns"], 1791237555658403500)
        self.assertEqual(saved["route"], payload["route"])
        self.assertEqual(saved["events"][0]["status"], "accepted")

    def test_timestamp_compatibility_still_rejects_changes_to_original_data(self):
        project = project_fixture()
        project["source"]["mtime_ns"] = 1791237555658403500
        atomic_json(self.path, project)
        store = ProjectStore(self.path)
        for key, value in (("mtime_ns", 1791237555658503600), ("sha256", "other"),
                           ("name", "other.MOV"), ("mtime_ns", "invalid")):
            payload = store.snapshot()
            payload["source"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                store.save(payload)


class ReviewServerTests(ReviewProjectTests):
    def setUp(self):
        super().setUp()
        self.server = ReviewServer(self.path, 0)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def request(self, path, method="GET", payload=None, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.url + path, data=data, method=method,
                          headers=headers or ({"Content-Type": "application/json"} if data else {}))
        return urlopen(request)

    def test_api_saves_reopens_and_rejects_stale_revision(self):
        with self.request("/api/project") as response:
            project = json.load(response)["project"]
        project["events"][0]["status"] = "rejected"
        with self.request("/api/project", "PUT", project) as response:
            self.assertEqual(json.load(response)["project"]["revision"], 1)
        with self.assertRaises(HTTPError) as error:
            self.request("/api/project", "PUT", project)
        self.assertEqual(error.exception.code, 409)

    def test_cross_origin_requests_and_traversal_are_rejected(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/api/project", headers={"Origin": "https://external.example"})
        self.assertEqual(error.exception.code, 403)
        with self.assertRaises(HTTPError) as error:
            self.request("/files/%2e%2e/outside.json")
        self.assertEqual(error.exception.code, 404)

    def test_video_range_and_missing_photo(self):
        (self.path.parent / "preview.mp4").write_bytes(b"0123456789")
        with self.request("/files/preview.mp4", headers={"Range": "bytes=2-5"}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
        with self.assertRaises(HTTPError) as error:
            self.request("/files/missing.jpg")
        self.assertEqual(error.exception.code, 404)

    def test_export_endpoint_rejects_pending_review(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/api/export", "POST", {})
        self.assertEqual(error.exception.code, 400)
