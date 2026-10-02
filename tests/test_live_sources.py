import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from src.config import RoadDamageConfig
from src.live_location import WindowsLocation
from src.live_session import LiveSession
from src.live_sources import AutomaticLocation, PhoneLocation, SimulatedLocation
from test_live import FixedRoad, FixedTracker, update_location


def payload(**changes):
    return dict({'_type': 'location', 'lat': -33.5, 'lon': -56.9,
                 'acc': 8, 'tst': time.time()}, **changes)


class SourceTests(unittest.TestCase):
    def test_phone_requires_real_accuracy_fresh_timestamp_and_ignores_delayed_retries(self):
        phone = PhoneLocation()
        message = payload()
        phone.ingest_phone(message)
        state, fix = phone.snapshot()
        self.assertTrue(state['valid'])
        self.assertEqual(fix.source, 'phone')
        phone.ingest_phone(payload(tst=message['tst'] - 20, lat=20))
        self.assertEqual(phone.snapshot()[1], fix)
        self.assertFalse(phone.snapshot(now=message['tst'] + 6)[0]['valid'])
        for message in (payload(acc=0), payload(acc=float('nan')), payload(lat=100), payload(tst=time.time()+100)):
            with self.assertRaises(ValueError):
                phone.ingest_phone(message)
        message = payload()
        del message['acc']
        with self.assertRaises(ValueError):
            phone.ingest_phone(message)

    def test_phone_endpoint_authentication_and_actual_owntracks_post(self):
        phone = PhoneLocation(port=0, host='127.0.0.1')
        phone.start()
        self.addCleanup(phone.close)
        url = f'http://127.0.0.1:{phone.port}'
        def post(path, data):
            return urlopen(Request(url + path, json.dumps(data).encode(),
                                   {'Content-Type': 'application/json'}, method='POST'), timeout=5)
        with self.assertRaises(HTTPError) as error:
            post('/phone/bad-token', payload())
        self.assertEqual(error.exception.code, 403)
        with post(phone.endpoint_path, payload()) as response:
            self.assertEqual(json.load(response), [])
        self.assertTrue(phone.snapshot()[0]['valid'])
        with urlopen(url + phone.endpoint_path, timeout=5) as response:
            status = json.load(response)
        self.assertTrue(status['online'])
        self.assertEqual(status['messages_received'], 1)
        self.assertEqual(status['last_message_type'], 'location')
        self.assertNotIn('latitude', json.dumps(status))
        with post(phone.endpoint_path, {'_type': 'transition'}) as response:
            self.assertEqual(response.status, 200)
        with self.assertRaises(HTTPError):
            post(phone.endpoint_path, payload(acc=0))
        self.assertIsNotNone(phone.receiver_status()['last_error'])

    def test_auto_waits_then_phone_then_simulation_and_locks_source_for_trip(self):
        windows = WindowsLocation()
        update_location(windows, accuracy=111)
        phone = PhoneLocation()
        auto = AutomaticLocation(windows, phone)
        with patch('src.live_sources.time.monotonic', return_value=100):
            self.assertEqual(auto.snapshot()[0]['source'], 'windows')
        with patch('src.live_sources.time.monotonic', return_value=109):
            self.assertEqual(auto.snapshot()[0]['source'], 'phone')
            self.assertFalse(auto.snapshot()[0]['valid'])
            phone.ingest_phone(payload())
            auto.begin_trip()
        update_location(windows)
        self.assertEqual(auto.snapshot()[0]['source'], 'phone')
        self.assertFalse(auto.snapshot(now=time.time()+6)[0]['valid'])  # Pause, don't fabricate GPS.
        auto.end_trip()
        self.assertEqual(auto.snapshot()[0]['source'], 'windows')
        update_location(windows, accuracy=111)
        phone.fix = None
        auto.bad_since = 100
        with patch('src.live_sources.time.monotonic', return_value=119):
            self.assertEqual(auto.snapshot()[0]['source'], 'mock')
            auto.begin_trip()
        phone.ingest_phone(payload())
        self.assertEqual(auto.snapshot()[0]['source'], 'mock')
        auto.end_trip()
        self.assertEqual(auto.snapshot()[0]['source'], 'phone')

    def test_no_gps_does_not_start_windows_or_phone_and_starts_immediately(self):
        auto = AutomaticLocation(mode='mock')
        with patch.object(auto.windows, 'start') as windows, patch.object(auto.phone, 'start') as phone:
            auto.start()
            windows.assert_not_called()
            phone.assert_not_called()
        state, fix = auto.snapshot()
        self.assertTrue(state['valid'])
        self.assertIsNone(state['accuracy_m'])
        self.assertEqual(fix.source, 'mock')

    def test_simulated_trip_map_and_every_evidence_are_labeled_and_follow_video_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'weights.pt'
            weights.touch()
            location = SimulatedLocation()
            config = RoadDamageConfig(enabled=True, model=weights, frame_interval=1, roi=(0,0,1,1), road_area=None)
            with patch('src.live_session.ObjectTracker', FixedTracker), patch('src.live_session.RoadDamageDetector', FixedRoad):
                session = LiveSession(location, root, road_config=config, desktop=True)
                session.start()
                try:
                    for _ in range(4):
                        session.submit(np.full((832,464,3), 100, np.uint8), time.time()).result(10)
                        time.sleep(.04)
                    session.finish()
                    self.assertTrue(session.done.wait(10))
                    summary = json.loads((session.directory / 'recorrido.json').read_text())
                    self.assertTrue(summary['complete'], summary['export_errors'])
                    self.assertTrue(summary['location']['simulated'])
                    events = json.loads((session.directory / 'recorrido_events.json').read_text())['events']
                    self.assertTrue(events)
                    self.assertTrue(all(e['location_source'] == 'mock' and e['location_accuracy_m'] is None for e in events))
                    self.assertIn('RUTA SIMULADA', (session.directory / 'map.html').read_text(encoding='utf-8'))
                    self.assertAlmostEqual(session.route.samples[0].latitude, -33.5145)
                    self.assertAlmostEqual(session.route.samples[-1].latitude, -33.5195)
                    self.assertTrue((session.directory / 'recorrido.mp4').is_file())
                finally:
                    session.close()


if __name__ == '__main__':
    unittest.main()
