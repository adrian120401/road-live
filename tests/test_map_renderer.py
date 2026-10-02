"""HTML generation, confidence filtering and portable evidence links."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.location import MockRouteLocationProvider, VideoMetadataLocationProvider, RoutePoint
from src.map_renderer import render_map


class MapTests(unittest.TestCase):
    def test_custom_output_embeds_thumbnail_and_links_to_original_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "events/foto con espacios.jpg"
            evidence.parent.mkdir()
            Image.new("RGB", (2160, 3840), "gray").save(evidence)
            events_path, output = root / "events.json", root / "maps/custom/map.html"
            event = {"event_id": 1, "type": "pothole", "track_id": 9, "confidence": 0.91,
                     "timestamp": 3.5, "frame": 106, "latitude": -33.515, "longitude": -56.8973,
                     "location_source": "mock", "evidence_path": "events/foto con espacios.jpg"}
            rejected = {**event, "event_id": 2, "confidence": 0.43}
            summary = render_map(MockRouteLocationProvider(10), [event, rejected], events_path, output,
                                 complete=False, processed_seconds=4)
            html = output.read_text(encoding="utf-8")
            self.assertEqual(summary["geolocated_potholes"], 1)
            self.assertIn("RUTA SIMULADA", html)
            self.assertIn("Prueba parcial", html)
            self.assertIn("data:image/jpeg;base64,", html)
            self.assertIn("../../events/foto%20con%20espacios.jpg", html)
            self.assertIn("91%", html)
            self.assertEqual(summary["average_pothole_confidence"],.91)
            self.assertIn("INICIO",html)
            self.assertIn("FIN",html)
            self.assertNotIn("Recorrido GPS estimado",html)
            self.assertIn('<meta name="referrer" content="no-referrer-when-downgrade">', html)
            self.assertIn("referrerPolicy:'no-referrer-when-downgrade'", html)
            self.assertIn("if(location.protocol==='http:' || location.protocol==='https:')", html)
            self.assertIn("visor local de Urban Vision", html)
            self.assertNotIn("EVENTO #2", html)
            with Image.open(evidence) as image:
                self.assertEqual(image.size, (2160, 3840))

    def test_zero_events_and_missing_evidence_still_produce_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "empty.html"
            summary = render_map(MockRouteLocationProvider(10), [], root / "events.json", output,
                                 complete=True, processed_seconds=10)
            self.assertEqual(summary["geolocated_potholes"], 0)
            self.assertIn("Sin pozos confirmados", output.read_text(encoding="utf-8"))
            event = {"event_id": 1, "track_id": 1, "confidence": 0.5, "timestamp": 0, "frame": 1,
                     "latitude": -33.5, "longitude": -56.9, "location_source": "mock",
                     "evidence_image": "missing.jpg"}
            render_map(MockRouteLocationProvider(10), [event], root / "events.json", output,
                       complete=True, processed_seconds=10)
            self.assertIn("Imagen no disponible", output.read_text(encoding="utf-8"))

    def test_distance_only_appears_for_real_temporal_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/"real.html"
            provider = VideoMetadataLocationProvider([RoutePoint(0,-33.5,-56.9),RoutePoint(10,-33.501,-56.9)])
            render_map(provider,[],root/"events.json",output,complete=True,processed_seconds=10)
            text = output.read_text(encoding="utf-8")
            self.assertIn("Recorrido GPS estimado",text)
            self.assertIn("0.11 km",text)
            self.assertIn("GPS REAL",text)
