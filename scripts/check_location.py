"""Inspect the exact native location provider used by the app, without a browser."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.live_location import WindowsLocation


def main():
    parser = argparse.ArgumentParser(description='Diagnóstico de ubicación nativa de Windows (sin navegador)')
    parser.add_argument('--seconds', type=int, default=15, help='Tiempo de observación, de 1 a 120 segundos')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120:
        parser.error('--seconds debe estar entre 1 y 120')
    if sys.platform != 'win32':
        print('Ejecutá este diagnóstico con Python de Windows, fuera de WSL.')
        return 1
    location = WindowsLocation()
    location.start()
    print('API nativa: System.Device.Location.GeoCoordinateWatcher · precisión solicitada: High', flush=True)
    print('No utiliza geolocalización ni permisos del navegador. No imprime coordenadas.', flush=True)
    previous = None
    last = None
    try:
        until = time.monotonic() + args.seconds
        while time.monotonic() < until:
            last, _ = location.snapshot()
            signature = (last['status'], last['permission'], last['accuracy_m'], last['valid'], last['reason'])
            if signature != previous:
                print(json.dumps(last, ensure_ascii=False), flush=True)
                previous = signature
            time.sleep(.5)
        print('Resultado final: ' + json.dumps(location.snapshot()[0], ensure_ascii=False), flush=True)
        print('±100 m o más es incertidumbre de la posición, no un permiso denegado.', flush=True)
        print('Esta API no identifica si la posición proviene de GPS, Wi-Fi o IP: position_source=Unknown.', flush=True)
        print('Habilitar Ubicación no confirma que exista un receptor GPS/GNSS. Windows debe reconocer el receptor y su controlador.', flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        location.close()


if __name__ == '__main__':
    raise SystemExit(main())
