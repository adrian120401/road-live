"""Real HTTP/image/evidence flow with injected models; no weights or GPS required."""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.config import Detection, RoadDamageConfig
from src.live import LiveApplication, create_server
from src.live_location import LiveRoute, LocationFix, WindowsLocation
from src.live_session import LiveSession
from src.map_renderer import render_map
from src.road_damage import RoadAssociator


def update_location(location, *, accuracy=8, age=0, latitude=-33.5, longitude=-56.9):
    location.ingest({'status': 'Ready', 'permission': 'Granted', 'latitude': latitude,
                     'longitude': longitude, 'accuracy_m': accuracy,
                     'timestamp': datetime.fromtimestamp(time.time() - age, timezone.utc).isoformat()})


class FixedTracker:
    device = 'cpu'

    def __init__(self, config):
        self.model = SimpleNamespace(predictor=None)

    def warmup(self, frame):
        pass

    def update(self, frame):
        return [Detection(1, 'car', .9, (100, 200, 160, 300))]


class FixedRoad:
    def __init__(self, config, device, fps, width, height):
        self.associator = RoadAssociator(config, fps)

    def warmup(self, frame):
        pass

    def update(self, frame, number, timestamp=None):
        self.last = self.associator.update([Detection(None, 'pothole', .9, (230, 450, 270, 480))], number, timestamp)
        return self.last


class LocationTests(unittest.TestCase):
    def test_location_gate_rejects_old_inaccurate_denied_and_nonfinite_positions(self):
        location = WindowsLocation()
        for accuracy, age in ((26, 0), (8, 6), (float('nan'), 0)):
            update_location(location, accuracy=accuracy, age=age)
            self.assertFalse(location.snapshot()[0]['valid'])
        location.ingest({'status': 'Ready', 'permission': 'Denied'})
        self.assertFalse(location.snapshot()[0]['valid'])
        update_location(location)
        self.assertTrue(location.snapshot()[0]['valid'])

    def test_repeated_cached_position_does_not_reset_age(self):
        location = WindowsLocation()
        epoch = time.time()
        row = {'status': 'Ready', 'latitude': -33, 'longitude': -56, 'accuracy_m': 8,
               'timestamp': datetime.fromtimestamp(epoch, timezone.utc).isoformat()}
        location.ingest(row)
        location.ingest(row)
        self.assertFalse(location.snapshot(epoch + 6)[0]['valid'])

    def test_segments_do_not_bridge_unrecorded_distance_and_empty_map_is_valid(self):
        route = LiveRoute()
        for t, lat in ((0, -33.5), (1, -33.501)):
            route.append(t, LocationFix(lat, -56.9, 8, 0))
        route.pause()
        for t, lat in ((10, -34.5), (11, -34.501)):
            route.append(t, LocationFix(lat, -56.9, 8, 0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_map(route, [], root / 'events.json', root / 'map.html', complete=True, processed_seconds=11)
            document = (root / 'map.html').read_text(encoding='utf-8')
            self.assertIn('0.22 km', document)
            self.assertIn('UBICACIÓN DE WINDOWS', document)
            render_map(LiveRoute(), [], root / 'events.json', root / 'empty.html', complete=False, processed_seconds=0)
            self.assertIn('Sin posiciones registradas', (root / 'empty.html').read_text(encoding='utf-8'))
        with self.assertRaises(ValueError):
            route.point_at(5)  # No position may be invented in a pause.

    def test_road_tracks_expire_by_capture_time_instead_of_processing_frame_count(self):
        associator = RoadAssociator(RoadDamageConfig(), 30)
        detection = Detection(None, 'pothole', .9, (20, 30, 50, 60))
        first = associator.update([detection], 1, timestamp=0)
        second = associator.update([detection], 2, timestamp=1)
        self.assertNotEqual(first.observed[0].track_id, second.observed[0].track_id)
        self.assertFalse(second.confirmed_ids)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        weights = self.root / 'weights.pt'
        weights.touch()
        self.road_config = RoadDamageConfig(enabled=True, model=weights, frame_interval=1)
        self.location = WindowsLocation()
        update_location(self.location)
        _, encoded = cv2.imencode('.jpg', np.full((832, 464, 3), 100, np.uint8))
        self.image = encoded.tobytes()
        self.patches = [patch('src.live_session.ObjectTracker', FixedTracker),
                        patch('src.live_session.RoadDamageDetector', FixedRoad)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.sessions = []
        self.addCleanup(self.close_sessions)

    def close_sessions(self):
        for session in self.sessions:
            if session.thread.is_alive():
                session.close()

    def session(self):
        session = LiveSession(self.location, self.root / 'outputs', device='cpu', road_config=self.road_config)
        session.start('Webcam USB')
        self.sessions.append(session)
        return session

    def frame(self, session):
        update_location(self.location)
        return session.submit(self.image, time.time()).result(timeout=10)

    def finish(self, session, reason='user_finished'):
        session.finish(reason)
        self.assertTrue(session.done.wait(10), 'Session did not finalize')
        return json.loads((session.directory / 'recorrido.json').read_text(encoding='utf-8'))

    def test_normal_finish_exports_last_evidence_real_position_and_readable_preview(self):
        session = self.session()
        self.assertIsNone(self.frame(session))  # Warmup
        result = self.frame(session)
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR))
        self.frame(session)
        report = self.finish(session)
        self.assertTrue(report['complete'])
        self.assertEqual(report['counts']['car'], 1)
        payload = json.loads((session.directory / 'recorrido_events.json').read_text(encoding='utf-8'))
        event = payload['events'][0]
        self.assertEqual(event['location_source'], 'windows')
        self.assertEqual(event['location_accuracy_m'], 8)
        self.assertTrue((session.directory / event['evidence_image']).is_file())
        self.assertTrue((session.directory / 'map.html').is_file())
        session.finish()
        self.assertEqual(session.snapshot()['state'], 'finished')

    def test_missing_location_blocks_start_and_pause_does_not_count_or_bridge_route(self):
        self.location.ingest({'permission': 'Denied'})
        with self.assertRaises(ValueError):
            self.session()
        update_location(self.location)
        session = self.session()
        self.frame(session)
        self.frame(session)
        self.frame(session)
        count = session.frames
        self.location.ingest({'permission': 'Denied'})
        self.assertIsNone(session.submit(self.image, time.time()).result(timeout=10))
        self.assertEqual(session.state, 'paused')
        self.assertEqual(session.frames, count)
        update_location(self.location, latitude=-34.5)
        session.submit(self.image, time.time()).result(timeout=10)
        self.assertEqual(len(session.route.segments), 2)
        self.assertEqual(session.analytics.counts['car'], 2)  # IDs start a new segment.
        report = self.finish(session)
        self.assertEqual(len(report['location']['segments']), 2)

    def test_disconnect_and_finish_without_frames_preserve_partial_empty_map(self):
        session = self.session()
        report = self.finish(session, 'camera_disconnected')
        self.assertFalse(report['complete'])
        self.assertEqual(report['frames_processed'], 0)
        self.assertIsNotNone(session.snapshot()['map_url'])

    def test_model_validation_failure_is_terminal_and_bad_jpeg_is_recoverable(self):
        session = self.session()
        with self.assertRaises(ValueError):
            session.submit(b'not jpeg', time.time()).result(timeout=10)
        self.assertFalse(session.done.is_set())
        self.frame(session)
        with patch.object(session.tracker, 'update', side_effect=ValueError('Injected model failure')):
            with self.assertRaisesRegex(ValueError, 'Injected model failure'):
                self.frame(session)
            self.assertTrue(session.done.wait(10))
        self.assertEqual(session.state, 'error')
        report = json.loads((session.directory / 'recorrido.json').read_text(encoding='utf-8'))
        self.assertFalse(report['complete'])
        self.assertIn('Injected model failure', report['error'])

    def test_finish_during_inference_saves_accepted_frame_and_rejects_new_uploads(self):
        session = self.session()
        self.frame(session)
        self.frame(session)
        entered, release = Event(), Event()
        original = session.tracker.update
        def blocked(frame):
            entered.set()
            release.wait(5)
            return original(frame)
        with patch.object(session.tracker, 'update', side_effect=blocked):
            future = session.submit(self.image, time.time())
            self.assertTrue(entered.wait(5))
            session.finish()
            with self.assertRaises(ValueError):
                session.submit(self.image, time.time())
            release.set()
            future.result(timeout=10)
            self.assertTrue(session.done.wait(10))
        self.assertEqual(session.frames, 2)
        self.assertEqual(session.road_analytics.count, 1)

    def test_lost_client_is_finalized_without_manual_button(self):
        session = self.session()
        session.last_contact = time.monotonic() - 11
        self.assertTrue(session.done.wait(10))
        self.assertEqual(session.stop_reason, 'client_disconnected')
        self.assertTrue((session.directory / 'map.html').is_file())

    def test_evidence_failure_preserves_event_position_summary_and_map(self):
        session = self.session()
        self.frame(session)
        self.frame(session)
        self.frame(session)
        with patch('src.road_analytics.cv2.imwrite', return_value=False):
            report = self.finish(session)
        self.assertFalse(report['complete'])
        self.assertEqual(report['road_damage']['potholes'], 1)
        self.assertEqual(session.state, 'error')
        self.assertTrue(session.snapshot()['map_url'])
        event = json.loads((session.directory / 'recorrido_events.json').read_text(encoding='utf-8'))['events'][0]
        self.assertEqual(event['latitude'], -33.5)

    def test_map_failure_keeps_evidence_and_summary_accessible(self):
        session = self.session()
        self.frame(session)
        self.frame(session)
        self.frame(session)
        with patch('src.live_session.render_map', side_effect=OSError('Map disk failure')):
            report = self.finish(session)
        self.assertFalse(report['complete'])
        self.assertEqual(report['road_damage']['saved_evidence'], 1)
        self.assertIsNone(session.snapshot()['map_url'])
        self.assertTrue(session.snapshot()['summary_url'])

    def test_http_camera_to_map_flow_and_output_paths(self):
        app = LiveApplication(self.location, self.root / 'http', 'cpu', self.road_config)
        server, url = create_server(app, 0)
        thread = Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        self.addCleanup(app.close)
        def post(path, data=b'', headers=None):
            return urlopen(Request(url + path, data=data, headers=headers or {}, method='POST'), timeout=10)
        with post('api/start', b'{"camera_name":"USB"}') as response:
            session_id = json.load(response)['id']
        with self.assertRaises(HTTPError) as conflict:
            post('api/start')
        self.assertEqual(conflict.exception.code, 400)
        for _ in range(3):
            with post(f'api/{session_id}/frames', self.image,
                      {'Content-Type': 'image/jpeg', 'X-Capture-Time': str(time.time() * 1000)}) as response:
                self.assertIn(response.status, (200, 204))
        with post(f'api/{session_id}/finish') as response:
            self.assertEqual(response.status, 202)
        self.assertTrue(app.session.done.wait(10))
        with urlopen(url + 'api/state') as response:
            state = json.load(response)['session']
        with urlopen(url.rstrip('/') + state['map_url']) as response:
            document = response.read()
            self.assertIn(b'data:image/jpeg;base64,', document)
            self.assertEqual(response.headers['Referrer-Policy'], 'no-referrer-when-downgrade')
        with self.assertRaises(HTTPError):
            urlopen(url + 'outputs/%2e%2e/weights.pt')
        with self.assertRaises(HTTPError) as forbidden:
            post('api/start', headers={'Origin': 'https://example.com'})
        self.assertEqual(forbidden.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
