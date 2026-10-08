# Analizar y revisar el recorrido

Desde PowerShell en la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-offline.txt
.\.venv\Scripts\python.exe -m src.offline serve --project outputs/recorrido/project.json
```

Se abre el editor local en `http://127.0.0.1:8765/`. Dejá la terminal abierta mientras
trabajás. Si ese puerto está ocupado, agregá `--port 0` para elegir uno disponible.
También podés abrir **Revisar recorrido.cmd** con doble clic; elige un puerto libre.

## Dibujar las calles y revisar las fotos

1. En **Dibujar recorrido**, hacé clic en el inicio y después en las esquinas o giros,
   en el orden en que pasaste. Agregá puntos para seguir una curva. El último punto es el fin.
2. Arrastrá los puntos para corregirlos. Seleccioná un punto para eliminarlo, o usá **Deshacer**.
3. En **Ubicar pozos**, seleccioná una detección. La foto muestra el detalle; podés abrir
   la foto completa y consultar ese momento en el video.
4. Elegí **Ubicar en el mapa** y hacé clic donde estaba el pozo. Eso lo confirma.
   Podés arrastrar el marcador o volver a pendiente. **Descartar** lo excluye del resultado.
5. Esperá **Guardado en tu equipo**. La ruta y la revisión se recuperan al reabrir el editor.
6. Cuando no queden pendientes y haya una ruta, **Exportar video 4K** crea el MP4 y el mapa.
   La exportación usa una copia de la revisión guardada al iniciar; editar después crea otra revisión.

La ruta se dibuja manualmente: no calcula calles automáticamente ni estima el avance
del vehículo. Las coordenadas no se presentan como GPS. La confianza del detector no
certifica un pozo real: tapas de saneamiento, parches y bordes rotos pueden confundirse.
Esta versión permite ubicar o descartar detecciones; no agregar pozos ni unir duplicados.

El mapa de calles necesita internet. La edición y las evidencias permanecen accesibles
si falla el fondo. Para exportar el mapa al video, las calles deben cargar correctamente.

## Archivos

Todo queda en `outputs/recorrido/`, separado del original:

- `analysis.json`, `analysis_events.json` y `observations.jsonl`: resultados originales.
- `analysis_events/`: fotos originales anotadas y detalles de cada detección.
- `project.json`: recorrido y revisión manual, con versión y revisión incremental.
- `preview.mp4`: copia liviana en H.264 para revisar en el navegador, sin audio.
- `source_sdr.mp4`: preparación SDR en 4K si el original es HDR/Dolby Vision.
- `recorrido_final_r<N>.mp4`: resultado 2160×3840, 60 FPS, sin audio, con ocho segundos de mapa al cierre.
- `reviewed_r<N>_map.html`, `reviewed_r<N>_map.png`, `reviewed_r<N>.geojson` y
  `reviewed_r<N>_events.json`: mapa y datos de la revisión exportada.
- `export_r<N>.json` y `.log`: verificación de salida y diagnóstico de FFmpeg.

El original nunca se reemplaza. Las exportaciones se versionan por revisión. Una salida
parcial queda con `.partial.mp4` y no se ofrece como resultado terminado.
La exportación conserva la duración usando los tiempos de los fotogramas, repitiendo
o suprimiendo frames solo para adaptar la reproducción a 60 FPS constantes.

## Analizar otro archivo o repetir la comparación

```powershell
.\.venv\Scripts\python.exe -m src.offline calibrate --input recorrido.MOV --output outputs/nuevo_recorrido/calibration --device cuda
```

Compara 1080p/640/intervalo 2, 4K/640/intervalo 2, 4K/960/intervalo 1 y 4K/1280/intervalo 1
en cinco fragmentos de un segundo. Antes de comparar, convierte HDR a SDR si corresponde.
El informe `comparison.json` incluye evidencias, eventos y tiempos. Seleccioná un perfil
del informe y guardá ese objeto JSON como `selected_profile.json`; la elección debe
considerar las fotos, no solo el número de detecciones. Los tiempos 22, 39, 53, 72 y 92 s
están elegidos para este recorrido; para videos diferentes adaptá `starts` en
`src/offline_calibration.py` o llamá `calibrate` con los tiempos correspondientes.

```powershell
.\.venv\Scripts\python.exe -m src.offline analyze --input recorrido.MOV --output outputs/nuevo_recorrido --profile outputs/nuevo_recorrido/selected_profile.json --device cuda
.\.venv\Scripts\python.exe -m src.offline serve --project outputs/nuevo_recorrido/project.json
```

Una carpeta con `project.json` u `observations.jsonl` no se vuelve a analizar encima.
Usá una carpeta distinta para conservar la revisión y los resultados anteriores.
`--device auto` puede recurrir a CPU; `cuda` exige una GPU compatible.

## Exportación y navegador

La exportación usa FFmpeg incluido por `imageio-ffmpeg`. Puede usarse uno propio con
`URBAN_VISION_FFMPEG`. Para capturar el mapa se usa Chrome de Windows si está instalado.
Si no, instalá Chromium para Playwright:

```powershell
.\.venv\Scripts\python.exe -m playwright install chromium
```

`URBAN_VISION_CHROME` permite indicar otro ejecutable de Chrome/Chromium.
El encoder final es H.264, CRF 16, preset slow; la conversión HDR intermedia usa
CRF 14 y preset fast. El procesamiento y la exportación 4K pueden tardar varios minutos.
También podés exportar una revisión ya terminada desde la terminal:

```powershell
.\.venv\Scripts\python.exe -m src.offline export --project outputs/recorrido/project.json
```

## Agregar proximidad sin volver a marcar el mapa

El proyecto conserva las calles dibujadas y cada pozo confirmado o descartado. Se puede
agregar la capa de **VEHÍCULO DELANTE / PRECAUCIÓN / MUY CERCA** al análisis guardado:

```powershell
.\.venv\Scripts\python.exe -m src.offline add-proximity --project outputs/recorrido/project.json --profile outputs/recorrido/proximity_recorrido.json --device cuda
.\.venv\Scripts\python.exe -m src.offline export --project outputs/recorrido/project.json --reuse-map --output recorrido_final_r33_precaucion.mp4
```

Esto analiza profundidad en la copia original SDR, reutiliza las detecciones generales,
y guarda la capa en `observations_proximity.jsonl` y `scene_layers.json`. No cambia
`project.json`, las posiciones manuales, los IDs de pozos ni el análisis original.
Las siguientes exportaciones del editor incorporan la capa adicional automáticamente.
`--reuse-map` usa la imagen exacta del cierre de esa revisión, sin recapturar las calles.
El archivo final anterior se conserva. Para otro recorrido, calibrar un perfil propio:
los umbrales son relativos y el aviso no representa metros ni una medición de seguridad.

Para activar la capa durante un análisis nuevo, `analyze` admite `--proximity-profile`.
El encoder CPU anterior continúa siendo el predeterminado. `export --encoder h264_nvenc`
permite usar NVIDIA en modo p7/CQ14 para una exportación de alta calidad en 4K/60.
