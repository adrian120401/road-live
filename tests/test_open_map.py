"""Real local HTTP origin, referrer policy and evidence paths for the map viewer."""

from pathlib import Path
import tempfile
import threading
import unittest
from urllib.parse import urljoin
from urllib.request import urlopen

from scripts.open_map import create_server


class MapViewerTests(unittest.TestCase):
    def test_http_origin_and_referrer_policy_keep_relative_evidence_accessible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "outputs/validation/map with spaces.html"
            map_path.parent.mkdir(parents=True)
            map_path.write_text("<html>Urban Vision</html>", encoding="utf-8")
            evidence = root / "outputs/events/photo.jpg"
            evidence.parent.mkdir()
            evidence.write_bytes(b"evidence")
            server, url = create_server(map_path, root, 0)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                self.assertIn("http://127.0.0.1:", url)
                self.assertIn("map%20with%20spaces.html", url)
                with urlopen(url) as response:
                    self.assertEqual(response.headers["Referrer-Policy"], "no-referrer-when-downgrade")
                    self.assertIn(b"Urban Vision", response.read())
                with urlopen(urljoin(url, "../events/photo.jpg")) as response:
                    self.assertEqual(response.read(), b"evidence")
            finally:
                server.shutdown()
                thread.join()
                server.server_close()

    def test_missing_map_and_map_outside_selected_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "map.html"
            with self.assertRaises(ValueError):
                create_server(path, root, 0)
            path.touch()
            with self.assertRaisesRegex(ValueError, "outside"):
                create_server(path, root / "other", 0)
