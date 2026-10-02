# Urban Vision

Prototipo local de percepción urbana con **YOLO26n**, **BoT-SORT**, IDs persistentes,
conteos aproximados de objetos únicos, trayectorias cortas y un HUD propio.
V2 agrega un segundo modelo local de **potholes**, eventos independientes y fotos de evidencia.
V3 agrega ubicación temporal real o simulada, mínimo de confianza del **50%** y mapa interactivo.
V3.1 suma revisión de cebras con YOLOE, profundidad YOLO26 y proximidad experimental,
con capas opcionales, métricas por etapa y mapa con evidencias ampliables.
Procesa video offline y ofrece una aplicación dedicada para recorridos con webcam en Windows;
no entrena modelos ni requiere servicios cloud para detección.

## Recorrido en vivo en Windows

Guía de instalación y uso: **[docs/WINDOWS.md](docs/WINDOWS.md)**.

Con las dependencias y los modelos instalados, ejecutar en PowerShell desde el proyecto:

```powershell
.\.venv\Scripts\python.exe -m src.live --device auto
```

Instalar la interfaz con `pip install -r requirements-desktop.txt` usando el Python del entorno.
Se abre una ventana **9:16**, con recorte central y el HUD original. **C** permite elegir
cámara antes de iniciar. Esperar ubicación nativa de Windows con precisión ≤25 m y
antigüedad ≤5 s y pulsar **Iniciar recorrido**, abajo. **Esc** o **F** finaliza y muestra
el mapa interactivo en la misma ventana. **Nuevo recorrido** vuelve a la cámara.

Cada sesión guarda `recorrido.mp4` (**1080×1920, 30 FPS, sin audio**), `map.html`,
`recorrido.json`, `recorrido_events.json` y fotos en `outputs/live/<fecha_id>/`.
La grabación comienza con la primera imagen analizada; repite imágenes si la inferencia
es más lenta para conservar la duración real. Los controles no aparecen en el MP4.
Al perder ubicación válida se pausa el registro y la grabación; se reanudan al recuperarla.
Las evidencias incluyen su posición temporal en el clip (`video_timestamp`).

La captura y la ubicación son nativas de Windows. Una laptop puede entregar ubicación
por Wi-Fi/IP: si no alcanza los requisitos, no podrá iniciar. El fondo de calles requiere
internet; detección, ruta y evidencias se procesan localmente.
`--output-root` cambia la carpeta de sesiones. `--road-roi LEFT TOP RIGHT BOTTOM` y
`--no-road-area` ajustan la zona de detección a la cámara. Desconectar la cámara,
cerrar con Alt+F4 o pulsar Ctrl+C conserva un resultado parcial.

Si la ubicación supera ±25 m, ejecutar este diagnóstico independiente del navegador:

```powershell
.\.venv\Scripts\python.exe scripts/check_location.py --seconds 15
```

Informa permisos del servicio nativo, margen de error y antigüedad de la posición.
`Granted` con ±100 m o más significa que hay permiso pero la precisión es insuficiente.
Esta API no identifica el proveedor físico: no se puede deducir GPS, Wi-Fi o IP del
margen de error. Activar ubicación o dar permiso al navegador no confirma que la laptop
tenga un receptor GPS/GNSS. Ver [diagnóstico en Windows](docs/WINDOWS.md#diagnosticar-la-ubicación).

## Instalación

Python 3.11 recomendado. En PowerShell, desde este directorio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe scripts/download_road_model.py
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

La variante CUDA 11.8 de PyTorch se eligió para la GTX 1050 Ti (Pascal).
No hace falta instalar el CUDA Toolkit ni cambiar el driver si la prueba CUDA funciona.
Para una instalación exclusivamente CPU, reemplazar `cu118` por `cpu` en ese comando.
Los pesos `yolo26n.pt` se descargan de Ultralytics en la primera ejecución; luego puede
procesarse offline. Las versiones probadas quedan fijadas en `requirements.txt`.

Si pip hereda índices externos que fallan, el archivo local `.venv/pip.ini` puede contener:

```ini
[global]
index-url = https://pypi.org/simple
extra-index-url =
```

## Ejecutar

En **zsh / bash / WSL**, usando el entorno Windows existente de este proyecto:

```bash
./.venv/Scripts/python.exe src/main.py --input video1.mp4 --show
```

En Linux con un entorno Python nativo, usar `.venv/bin/python`. No usar barras invertidas
de PowerShell en zsh. Los siguientes comandos son para **PowerShell**:

```powershell
.\.venv\Scripts\python.exe src/main.py --input video1.mp4
.\.venv\Scripts\python.exe src/main.py --input video1.mp4 --show
.\.venv\Scripts\python.exe src/main.py --input video1.mp4 --conf 0.15 --output outputs/prueba.mp4
.\.venv\Scripts\python.exe src/main.py --input video1.mp4 --device cpu --tracker bytetrack.yaml
```

Con el entorno activado, también funciona `python src/main.py --input video1.mp4`.
Se admiten `--model` (modelo de detección), `--tracker` (YAML de Ultralytics o propio),
`--conf`, `--output`, `--device auto|cpu|cuda` y `--show`.
Las rutas relativas se resuelven desde el directorio donde se ejecuta el comando.

- Video: `outputs/urban_vision_output.mp4`, sin audio, lado máximo 1920 y FPS de la fuente.
- Analytics: `outputs/urban_vision_output.json`, con métricas, conteos y estadísticas por ID.
- `Q`, Escape o cerrar la ventana detiene la preview y finaliza un video parcial.
- `Ctrl+C` durante el procesamiento también finaliza el archivo parcial.
- Un JSON con `complete: false` y código de salida 130 identifica una interrupción.
- CUDA se prueba realmente al iniciar; `auto` utiliza CPU si la GPU no es compatible.

El HUD normal no muestra FPS de procesamiento. Consola/JSON miden el pipeline completo,
excluyendo la descarga del modelo y su calentamiento. Debug distingue FPS de video e
inferencias de FPS de procesamiento. Un run a 4 FPS de procesamiento puede generar un
video que se reproduce a 30 FPS: se procesan todos los frames, con su duración original.
La exportación OpenCV utiliza `mp4v` y FPS constante: videos con FPS variable se normalizan
al promedio de origen. Para publicación directa en plataformas que exijan H.264, se puede
reexportar desde el editor usado para el Reel.

## Conteos y límites

Clases: person, car, motorcycle, bicycle, bus, truck, traffic light y stop sign.
**VEHICLES significa solo autos**; motos, buses y camiones tienen contadores separados.
El HUD muestra autos, personas, motos y bicicletas. El JSON y el log incluyen las ocho clases.

`stop sign` requiere confianza >= 0.65 para mostrarlo y contabilizarlo. La revisión del video
encontró señales amarillas de badén mal clasificadas como STOP con confianza máxima 0.61.
Este filtro conservador vive en `config.py`; el tracker sigue recibiendo candidatos débiles
para mantener asociaciones. Puede omitir señales reales poco claras y no garantiza corregir
todos los errores de clasificación del modelo.

Cada ID se cuenta una sola vez. La categoría utiliza votos ponderados por confianza de
las últimas diez observaciones y tres confirmaciones de mayoría; una corrección transfiere
el contador entre categorías sin agregar un objeto. Esto corrige casos donde la primera
clasificación errónea quedaba congelada, sin modificar el tracker. No se cuentan detecciones sin ID. Las trayectorias usan
el centro inferior de la caja y solo los últimos 30 frames, sin conectar intervalos de oclusión.
Son desplazamientos en la imagen, afectados por el movimiento del auto; no velocidades reales.

**Los conteos no son una identidad física perfecta**: perder y reasignar un ID puede inflarlos,
y un intercambio de IDs puede reducirlos. Analytics está separado del tracker para mejorar
esa lógica en V2. Con cámara en movimiento, `persist=True` conserva estado pero no garantiza
que todos los objetos mantengan su ID durante cualquier oclusión.

## Código y pruebas

`src/main.py` orquesta el video; `tracker.py` adapta Ultralytics; `analytics.py` mantiene
conteos/historial; `renderer.py` dibuja el HUD; `config.py` contiene defaults y registros tipados.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas verifican conteos por ID, categorías, reapariciones, resultados sin ID y límites
temporales del historial, argumentos CLI, fallback CPU y MP4 legible ante fallos de inferencia.
No descargan pesos ni necesitan CUDA.

Para auditar el video exportado y generar una hoja de frames de muestra:

```powershell
.\.venv\Scripts\python.exe scripts/verify_output.py
```

Si otra aplicación satura la GPU, `--device cpu` permite completar una exportación sin
competir por CUDA. Las mediciones de rendimiento dependen de esa carga concurrente.

## V2: potholes y evidencia

En zsh / WSL (en PowerShell reemplazar `./.venv/Scripts/python.exe` por `.\.venv\Scripts\python.exe`):

```bash
./.venv/Scripts/python.exe scripts/download_road_model.py
./.venv/Scripts/python.exe src/main.py --input video1.mp4 --road-damage
./.venv/Scripts/python.exe src/main.py --input video1.mp4 --road-damage --show --debug-road
./.venv/Scripts/python.exe src/main.py --input video1.mp4 --road-damage --road-interval 3 --road-conf 0.60
./.venv/Scripts/python.exe src/main.py --input video1.mp4 --road-damage --collect-road-candidates
```

El script descarga **YOLOv8s de PeterHdd** (22,5 MB), verifica SHA-256 y fija la revisión
`101b80987b3fa86f9747d4170f936760ebad3a9f`. La inferencia usa el archivo local
`models/pothole_yolov8s.pt`. No necesita nuevas dependencias ni una API de inferencia.

- [Modelo y resultados publicados](https://huggingface.co/peterhdd/pothole-detection-yolov8).
- [Código de entrenamiento](https://github.com/PeterHdd/pothole-detection-yolo).
- [Dataset declarado](https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8), un export de
  [Roboflow Potholes Detection v1](https://universe.roboflow.com/project-ssayl/potholes-detection-d4rma/dataset/1).

El dataset y checkpoint nombran su única clase `0`. **Solo para el checkpoint exacto con
checksum verificado** se normaliza a `pothole`. Otros modelos deben contener una clase
`pothole`, `potholes` o `D40`; se filtra esa clase aunque el modelo reconozca también cracks.
No se considera equivalente cualquier clase numérica ni se usan pesos COCO como detector de pozos.

Defaults en `RoadDamageConfig`: confidence **0.50**, entrada **640**, intervalo **2**,
ROI **`0.05 0.45 0.95 0.65`**, evidencias activadas, candidatos de dataset desactivados.
La ROI se expresa como izquierda, arriba, derecha y abajo normalizados. En `video1.mp4`
corresponde a `(23, 374, 441, 541)` píxeles: la mitad inferior completa incluiría mucho tablero.
Se procesa el recorte y se restauran las coordenadas al frame original. Para otra cámara:

```bash
./.venv/Scripts/python.exe src/main.py --input otro.mp4 --road-damage --road-roi 0.05 0.45 0.95 0.70
```

Con `--road-damage`, el output predeterminado es `outputs/urban_vision_v2_output.mp4`.
La ejecución sin esa opción conserva el pipeline y output de V1.1. `--output` admite otra ruta.

Cada evento confirmado tiene un ID independiente y se cuenta una sola vez. Se confirma
después de dos observaciones consecutivas del detector; los frames intermedios reutilizados
no generan observaciones ni eventos. Asociación uno a uno por IoU/proximidad, con memoria
de hasta 0.5 segundos para pérdidas breves. Un pozo puede volver a contarse después de
una pérdida larga: es un conteo estimado, sin identidad geográfica.
El segundo modelo usa NMS con IoU **0.45**: en la primera revisión, el default 0.70
retenía dos propuestas superpuestas del mismo daño y generaba eventos duplicados.

Archivos generados (nombres derivados de `--output`):

- `*_events.json`: eventos con tipo, ID, confianza, bbox, frame, timestamp y ruta de foto.
- `*_events/pothole_001.jpg`: una foto anotada de la mejor observación por evento,
  guardada al expirar el track o al finalizar, incluyendo interrupciones.
- `*.json`: analytics generales y sección `road_damage` con conteos y métricas separadas.
- Con `--collect-road-candidates`: originales y metadatos `unreviewed` en
  `dataset_candidates/potholes/unreviewed/<output_stem>/`. No se generan etiquetas de entrenamiento.

`--no-road-evidence` desactiva las fotos. El timestamp usa los tiempos PTS del video cuando
son válidos y crecientes, con fallback a `(frame - 1) / FPS`; no es tiempo de procesamiento.
La mejor observación puede preceder al frame
de confirmación del evento. Las fotos **son evidencia de una detección**, no certifican un daño real.
Las ejecuciones con el mismo output reemplazan sus archivos; usar nombres diferentes para
conservar recorridos. Revisar los eventos del JSON de la ejecución actual al consultar fotos.

El HUD conserva el estilo existente y agrega una fila POZOS y cajas ámbar discretas.
`--debug-road` dibuja ROI y resultados antes de confirmación temporal, y registra scores,
cantidad de llamadas y tiempo por llamada. Los resultados raw ya están filtrados por `--road-conf`.
En modo normal solo se muestran eventos confirmados y métricas finales.

Los tiempos `mean_inference_ms` / `inference_fps` vienen del profiler de Ultralytics;
`mean_stage_ms` incluye pre/postprocesamiento y, para el general, tracking. El FPS total incluye
lectura, HUD, exportación y finalización de evidencias. Warmup está excluido de las estadísticas.

Para comparar intervalos y auditar el output:

```bash
./.venv/Scripts/python.exe scripts/benchmark_road.py --frames 180 --device cuda
./.venv/Scripts/python.exe scripts/verify_output.py --video outputs/urban_vision_v2_output.mp4
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

La compatibilidad técnica del checkpoint no garantiza calidad local. Puede confundir
parches, sombras y juntas con pozos. El video actual tiene poca resolución y una franja
estrecha de pavimento; una nueva grabación debe mostrar más calle cercana, menos cielo
y tablero, con buena iluminación y daños claramente visibles durante varios frames.
RDD2022 es una referencia para una futura ampliación; esta V2 solo muestra potholes y
no incluye entrenamiento, GPS, mapas, segmentación ni alertas de conducción.

La validación del video actual está en
[`outputs/validation/urban_vision_v2_output/review.md`](outputs/validation/urban_vision_v2_output/review.md).
La exportación final obtuvo **32.46 FPS en CUDA**, frente a **44.75 FPS** del general,
con IDs y estadísticas generales idénticos. Hay **5 eventos y 5 fotos**: dos daños
compatibles con pozos, dos casos ambiguos y un falso positivo claro en un badén.
La revisión visual no equivale a ground truth. No hubo warnings de inferencia en las
pasadas finales; el video completo y las fotos pasaron la auditoría.

## Interfaz en español y videos originales del iPhone

El HUD, los boxes, el modo debug y las fotos de evidencia se muestran en español:
AUTO, PERSONA, MOTO, BICICLETA, ÓMNIBUS, CAMIÓN, SEMÁFORO, PARE y POZO.
Se mantiene el nombre Urban Vision. Los nombres internos de clases y las claves JSON
se conservan para mantener compatibilidad con modelos y analytics.
Pillow renderiza tildes y caracteres Unicode en pequeños recortes de texto cacheados;
no convierte el frame completo para dibujar cada etiqueta.

Los `.MOV` se leen directamente. OpenCV aplica la orientación del iPhone; la exportación
conserva las proporciones de los frames orientados y el FPS promedio de la fuente.
`video2.MOV` y `video3.MOV` son verticales **2160×3840** (4K), unos 30 FPS y 64 segundos.
El perfil actual reduce estos frames a **1080×1920 antes de todos los detectores**, tracking
y overlays. Video, evidencias y candidatos nuevos usan esa resolución; el original queda
intacto. `--processing-max-side 0` permite analizar/exportar nuevamente en 4K.
La recodificación MP4 de OpenCV no conserva el codec/HDR ni
el audio original. La lectura y exportación 4K son más lentas que con `video1.mp4`.

Para estas dos grabaciones, la calle cercana se extiende más abajo. Usar la ROI
`0.05 0.48 0.95 0.70`, sin modificar el default utilizado por el video anterior:

```bash
./.venv/Scripts/python.exe src/main.py --input video2.MOV --road-damage --road-roi 0.05 0.48 0.95 0.70 --output outputs/urban_vision_video2_es.mp4
./.venv/Scripts/python.exe src/main.py --input video3.MOV --road-damage --road-roi 0.05 0.48 0.95 0.70 --output outputs/urban_vision_video3_es.mp4
```

Los eventos y sus fotos utilizan el mismo nombre base que cada output. La auditoría
guarda frames de muestra en resolución de exportación y usa miniaturas para la hoja de contacto.

## V3: geolocalización y mapa

```bash
./.venv/Scripts/python.exe src/main.py --input video2.MOV --road-damage --road-roi 0.05 0.48 0.95 0.70 --output outputs/urban_vision_v3.mp4
# Forzar simulación, incluso si existe GPS temporal:
./.venv/Scripts/python.exe src/main.py --input video2.MOV --road-damage --force-mock-route --map-output outputs/map.html --output outputs/otra_prueba.mp4
# Omitir inspección GPS y usar simulación:
./.venv/Scripts/python.exe src/main.py --input video1.mp4 --road-damage --no-use-video-gps
```

`--use-video-gps` está activado por defecto. La inspección automática usa **ExifTool local**:
primero el `--exiftool` explícito, luego PATH y finalmente la copia portable Windows en
`tools/exiftool/`. Esta copia conserva su carpeta `exiftool_files`; origen y SHA256 están
en `tools/exiftool/SOURCE.md`. No se descargan herramientas durante la ejecución.
Para Linux/macOS, instalar ExifTool en el PATH según sus
[instrucciones oficiales](https://exiftool.org/install.html), o pasar `--exiftool /ruta/exiftool`.
Leaflet **1.9.4** se incluye en `assets/leaflet/` con su licencia; no se agregan dependencias Python.

El extractor usa `-ee3` y preserva los identificadores de muestras. Nunca combina una
coordenada global con tiempos de metadata de orientación, caras o iluminación del iPhone.
Se informa por separado si hay GPS, una coordenada única, un track insuficiente o si la
inspección falló/no estaba disponible. En todos esos últimos casos se usa **mock**, sin
interrumpir detección, video ni fotos. La fecha se conserva con su zona horaria.

Un track real requiere al menos dos tiempos distintos, coordenadas válidas, intervalos
de hasta 10 s y cobertura del video con tolerancia de 2 s en sus extremos. Estos límites
se configuran en `LocationConfig`. Se interpola entre puntos; dentro de la tolerancia
de los extremos se mantiene el punto más cercano. El mapa indica la posición aproximada
del **vehículo al detectar** el daño, no una posición física calculada del pozo.

Si no existe un track utilizable, `assets/trinidad_route.json` define un corredor aproximado
de 550 m junto a Francisco Fondar y Plaza Constitución. Se distribuye el avance por
distancia a lo largo de toda la duración del video, sin aleatoriedad ni servicios de routing.
Se exportan puntos cada segundo y el extremo final. Una prueba parcial muestra el recorrido
simulado completo y especifica cuántos segundos fueron procesados; los eventos corresponden
solo a esos frames. **No es una reconstrucción del recorrido real.**

El umbral de pozos es obligatorio: `--road-conf` admite **0.50 a 1.00**. Propuestas desde
0.35 se obtienen solo para medir descartes y se filtran antes de asociar IDs. No aparecen
en HUD/debug, analytics, candidatos de dataset, eventos, fotos ni mapas si están por debajo
del umbral elegido. `discarded_below_50_observations` cuenta observaciones entre 35% y 50%,
no pozos distintos; propuestas inferiores a 35% no se miden. Un umbral mayor puede omitir
daños reales y tampoco garantiza eliminar todos los falsos positivos.

Salidas adicionales, derivadas del nombre del video de salida:

- `*_events.json`: conserva las claves de V2 y agrega `event_id`, `latitude`, `longitude`,
  `location_source` (`real`/`mock`) y `evidence_image` (alias compatible de `evidence_path`).
  Las rutas de evidencia son relativas al JSON; timestamp, coordenadas y foto corresponden
  a la misma observación representativa.
- `*_map.html`: recorrido y marcadores, con popups en español, miniaturas incrustadas y
  enlaces a fotos originales. `--map-output` permite elegir otra carpeta.
- `*.json`: agrega reporte `location`, puntos temporales en `location.trajectory`,
  método de timestamps y descartes del detector vial. V3.1 no exporta GeoJSON.

Abrir el mapa con el **visor HTTP local** (la terminal debe permanecer abierta):

```bash
./.venv/Scripts/python.exe scripts/open_map.py --map outputs/validation/v3/map.html
```

El visor abre el navegador en `http://127.0.0.1:8765/...`. `--port 0` elige un puerto libre;
`--no-open` muestra la URL sin abrir el navegador. Solo sirve archivos locales y no actúa
como proxy de tiles ni agrega dependencias. Para un mapa fuera del proyecto, elegir
`--root` como directorio común del HTML y sus evidencias.

OpenStreetMap requiere un `Referer` web válido. Abrir un HTML directamente con `file://`
no puede proporcionarlo, aunque se configure una política de referrer. La V3 ahora evita
solicitar tiles en ese modo y muestra un aviso para usar el visor. La página y el visor usan
`Referrer-Policy: no-referrer-when-downgrade`, conservando la URL HTTP real de la página.
Ver [política oficial](https://operations.osmfoundation.org/policies/tiles/).
Las calles se cargan online; ruta, marcadores, datos y miniaturas están dentro del HTML
y funcionan sin conexión. Se conserva la atribución y el cache normal del navegador.
No se suben videos ni fotos al proveedor de mapas. Sin detecciones se muestra igualmente
el recorrido. Sin `--road-damage`, se mantiene el detector general y se genera un mapa sin
eventos. Las salidas anteriores de V2 no se modifican ni se consideran resultados de V3.

Validación de V3: prueba parcial de 360 frames / 12 s de `video2.MOV`, 4K vertical,
3 eventos confirmados (65.3%, 84.8%, 78.7%), 3 evidencias nativas y 15 observaciones
descartadas entre 35% y 50%. Procesamiento CUDA: **4.83 FPS**, reproducción: unos 30 FPS.
ExifTool encontró coordenada única (-33.5147, -56.8963) y fecha, sin muestras GPS temporales:
se usó ruta **mock** de 65 puntos. Estos son eventos del modelo, no pozos reales certificados.

Para la primera prueba con geolocalización real, grabar un video y un track GPS temporal
sincronizado, conservar el archivo original y verificar que la herramienta de captura
embeba muestras GPS: la metadata habitual de la cámara del iPhone contiene una ubicación
global y no garantiza un recorrido. La importación de un GPX externo sería una siguiente
extensión pequeña del proveedor, sin tocar detección/tracking.

## V3.1: percepción urbana y mapa mejorado

Preparar los pesos oficiales una sola vez (zsh/WSL):

```bash
./.venv/Scripts/python.exe scripts/download_depth_model.py
./.venv/Scripts/python.exe scripts/download_crosswalk_model.py
# Ejemplo visual revisado del cruce de video2; no entrena el modelo:
./.venv/Scripts/python.exe scripts/prepare_crosswalk_visual.py --input video2.MOV --frame 270 --box 12 2180 2155 2510
```

Estos scripts verifican SHA-256. El último guarda un embedding local y su procedencia;
la inferencia de video no descarga modelos, no usa un text encoder ni nuevas dependencias.
Para utilizar otra foto de referencia, pasar `--reference imagen.jpg --box X1 Y1 X2 Y2`.
Preparar referencias nuevas reemplaza el perfil: conservar los NPZ/JSON de cada experimento.

Procesar el video revisado con el perfil de proximidad medido para esta cámara:

```bash
./.venv/Scripts/python.exe src/main.py --input video2.MOV \
  --road-damage --road-roi 0.05 0.48 0.95 0.70 \
  --crosswalks --proximity --proximity-profile assets/proximity_video2.json \
  --collect-road-candidates --map-output outputs/map.html \
  --output outputs/urban_vision_reel_output.mp4
./.venv/Scripts/python.exe scripts/open_map.py --map outputs/map.html
./.venv/Scripts/python.exe scripts/verify_output.py --video outputs/urban_vision_reel_output.mp4
```

En PowerShell, usar `.\.venv\Scripts\python.exe` y escribir el comando en una línea.
Sin `--crosswalks` / `--proximity` estas capas no se cargan. `--no-traffic-lights` desactiva
semáforos en el general. `--no-enhanced-popups` admite una vista resumida del panel del mapa.
Los eventos geolocalizados continúan siendo exclusivamente los pozos.

### Perfil para Reel y aviso de proximidad

Defaults: `--processing-max-side 1920 --prefetch-frames 2`. El video actual se analiza y
exporta a **1080×1920**, sin recortar, saltear frames ni cambiar timestamps. Videos menores
no se amplían; otras proporciones se conservan y las dimensiones reducidas son pares.
Para comparación nativa/secuencial: `--processing-max-side 0 --prefetch-frames 0`.

Un único lector de OpenCV decodifica/reduce el próximo frame mientras se procesa el actual.
La cola tiene dos frames de espera; modelos/tracking siguen en el hilo principal. Los
paquetes conservan índice y PTS de origen. Interrupciones y errores cierran el lector y
finalizan las salidas parciales como antes. No agrega dependencias.

El banner central **VEHÍCULO DELANTE / PRECAUCIÓN** (ámbar) o **MUY CERCA** (naranja)
aparece solo para los estados confirmados del analizador existente. El track ID es secundario
y las esquinas del box resaltan el vehículo elegido. Mantiene persistencia/hysteresis y
no afirma distancia exacta ni riesgo de colisión. Debug muestra métricas; la demo normal
conserva frames/tiempo y oculta FPS de procesamiento.

JSON incluye `source_resolution`, `resolution`, `processing_max_side`, `prefetch_frames`,
`playback_fps` y los tiempos `decode`, `resize`, `reader_wait`, modelos/render/escritura y
`total_frame`. Con prefetch, lectura/resize se solapan con el procesamiento: sus tiempos
no se suman directamente al total. El archivo 4K anterior queda como referencia en
`outputs/urban_vision_v3_1_roadway_output.mp4`. El pipeline sigue siendo offline.

Validación completa de `video2.MOV`: **4,841 → 11,717 FPS de procesamiento (2,42×)**,
**395,57 → 163,44 segundos** de ejecución, los 1.915 frames exportados a 1080×1920 y
30,002 FPS de reproducción. Pasan 93 pruebas; se auditaron todos los frames, timestamps,
evidencias y coordenadas. El mapa contiene seis eventos con seis fotos correctamente
vinculadas; se comprobó en navegador de escritorio y móvil.

El cambio de resolución conserva 235 IDs generales, ocho semáforos y una cebra, pero
modifica algunos scores, categorías e IDs. De ocho eventos viales anteriores quedan seis:
el candidato ambiguo del frame 805 baja de 53% a 35,2%/48,7%; la cavidad observada en
1767 no obtiene la segunda confirmación (frame 1769 al 48%), pero se confirma en 1779
al 82,6%. No se bajó el mínimo del 50% para recuperarlos. Los casos de vereda 4/5/6/8
siguen excluidos. El perfil de proximidad se mantiene: mismos 13 cambios de estado y
desplazamiento máximo de tres frames. Comparación: `outputs/validation/reel/comparison.json`.

### Excluir daños en veredas

El detector vial ahora filtra propuestas **antes de asignar IDs** con una zona de calzada
independiente del corredor de proximidad. Un candidato necesita su centro dentro del
polígono y al menos **60% del área del box** dentro de él. Se mantiene confianza >=50%
y confirmación en dos inferencias. Las propuestas excluidas no llegan al HUD, contador,
eventos, evidencias, candidatos automáticos ni mapa.

El polígono predeterminado, revisado sobre `video2.MOV`, usa coordenadas normalizadas:
`0.48 0.48 0.72 0.48 1.0 0.70 0.0 0.70` (cuatro puntos X Y en orden). La ROI rectangular
continúa siendo el recorte de inferencia; el polígono decide qué resultados aceptar.
Es una aproximación conservadora para esta posición del iPhone: puede omitir daños
junto al cordón y requiere ajustar la zona si cambia la cámara, el ancho de calle o
el auto dobla. No reconoce semánticamente el límite entre asfalto y vereda.

```bash
# Ver recorte, calzada y propuestas rechazadas, únicamente en debug:
./.venv/Scripts/python.exe src/main.py --input video2.MOV --road-damage --road-roi 0.05 0.48 0.95 0.70 --debug-road
# Ajustar puntos y cobertura mínima sin modificar el detector:
./.venv/Scripts/python.exe src/main.py --input otro.mov --road-damage --road-area 0.48 0.48 0.72 0.48 1.0 0.70 0.0 0.70 --road-area-overlap 0.60
# Recuperar el comportamiento anterior para comparar otra posición de cámara:
./.venv/Scripts/python.exe src/main.py --input otro.mov --road-damage --no-road-area
```

El resumen JSON guarda el polígono y `discarded_outside_road_observations`: son
observaciones rechazadas con confianza suficiente, no un conteo de pozos únicos.
Los IDs de pozos pueden cambiar al eliminar candidatos antes del tracking. La revisión
de los eventos originales 4, 5, 6 y 8 usa sus frames y boxes, no una lista de IDs prohibidos.

Validación del filtro sobre los 1.915 frames de `video2.MOV`, conservando 2160×3840:
**14 → 8 eventos**, ocho evidencias y ocho marcadores. Los eventos originales 4/5/6/8
quedan excluidos durante toda su ventana temporal; también salen los candidatos laterales
1/3. Los pozos de calzada 11/12 conservan sus frames representativos y confidence.
Tracking general, conteos, cebras y transiciones de proximidad son idénticos al run anterior.
Rendimiento medido: **4,831 → 4,841 FPS**. Las 50 observaciones rechazadas por ubicación
son independientes de las 80 propuestas descartadas por confianza <50%.
Comparación y capturas: `outputs/validation/road_area/comparison.json` y
`outputs/validation/road_area/before_after_*.jpg`. Persisten confusiones del modelo con
grietas/juntas y posible fragmentación de IDs: ocho eventos no certifican ocho pozos físicos.

El mapa agrega duración, confianza promedio, fuente, INICIO/FIN y diferencias discretas
de opacidad entre confianza 50–69% y >=70%. Para mock, indica RUTA SIMULADA y no muestra
una distancia inventada. Cada popup reutiliza la evidencia original: detalle ampliado
incrustado, fotograma completo desplegable y enlace al archivo 4K. No crea nuevas fotos.
Las miniaturas y datos son portables; las calles de OpenStreetMap necesitan conexión y
el visor HTTP local, como en V3. No se exportan archivos GeoJSON nuevos.

**Cebras.** COCO no contiene esta clase. Se evaluó el modelo especializado
[YOLOv5s6 de kairess](https://github.com/kairess/crosswalk-traffic-light-detection-yolov5),
revisión `5ba1e01d946e1a5cc9dcf396b3b8fc62ae2666d5`, con conversión ONNX y paridad PyTorch/OpenCV.
En este video sus cajas resultaron incorrectas y la inferencia CPU costó unos 316 ms.
La alternativa elegida es [YOLOE-26s-seg oficial](https://docs.ultralytics.com/models/yoloe/),
con un ejemplo visual del frame 270. Las palabras por sí solas no confirmaban el cruce.
Esto **no demuestra generalización a otras calles**: se valida en la misma escena de referencia,
con controles negativos del video. No hubo entrenamiento. El modelo admite segmentación;
esta integración utiliza cajas discretas, sin pintar el pavimento completo.

YOLOE conserva el contexto completo de la imagen durante inferencia y exige que al menos
80% de la caja quede dentro de la ROI configurable `0 0.42 1 0.74`. El recorte de calle
antes de inferir empeoró los resultados revisados. Confianza de cebras: **0.35**;
NMS IoU **0.40** elimina propuestas solapadas del mismo cruce observadas en frames 271/274.
Inferencia cada tres frames y tres observaciones consecutivas para confirmar un ID;
no se confirma usando frames visuales reutilizados. Los IDs pueden fragmentarse tras oclusiones.

El exportador descartado sigue disponible opcionalmente, fuera de las dependencias del runtime:

```bash
./.venv/Scripts/python.exe -m pip install --no-deps --target tools/crosswalk-export-deps -r tools/crosswalk-export-requirements.txt
./.venv/Scripts/python.exe scripts/prepare_crosswalk.py
# Solo para revisar esa alternativa, no para la demo final:
./.venv/Scripts/python.exe src/main.py --input video2.MOV --crosswalks --crosswalk-model models/crosswalk_yolov5s6.onnx
```

**Proximidad.** [YOLO26n-depth](https://docs.ultralytics.com/tasks/depth/) procesa un frame reducido
con lado máximo 768, cada tres frames y solo si existe un vehículo candidato. Se muestrea
la mediana de la zona central/inferior de cada caja, después de descartar valores inválidos
y extremos. La muestra requiere 75% de píxeles válidos; se invalida al cambiar mucho la caja
o superar su tiempo de vigencia. Nunca se lee un único píxel ni se redimensiona un depth map a 4K.

El candidato debe ser AUTO/MOTO/ÓMNIBUS/CAMIÓN y tener centro y base dentro del trapecio
configurable. Cinco frames de presencia, dos muestras nuevas para subir de estado y tres
para bajar, mediana temporal e histéresis reducen oscilaciones. Cambiar o perder el ID reinicia
la persistencia. El HUD muestra **VEHÍCULO DELANTE / PRECAUCIÓN / CERCA**; no muestra metros.
Los semáforos mantienen la clase existente; esta versión no determina su estado.

`assets/proximity_video2.json` conserva los frames revisados y las distribuciones medidas:
SAFE 950–1000, CAUTION 1050–1110 y NEAR 1170–1250. Los umbrales se obtienen de medias
geométricas entre medianas, con histéresis del P90 de variaciones temporales. Son específicos
de este montaje, **no distancias de seguridad**. Un corredor fijo no determina el carril ni
si un vehículo está estacionado. La validación comprueba los candidatos y avisos en el video.
Sin `--proximity-profile`, registra profundidad pero no emite avisos. No reutilizar el perfil
si cambia cámara, resolución de inferencia, ángulo o corredor.

Opciones: `--crosswalk-conf`, `--crosswalk-interval`, `--crosswalk-roi`, `--depth-interval`,
`--depth-size`, `--road-corridor` (ocho valores X Y), `--no-depth-enabled`, `--debug-scene`.
`--display-conf 0.30` oculta cajas generales débiles, sin cambiar el umbral del tracker,
los IDs ni analytics; semáforos y PARE conservan sus gates existentes. Se eligió tras revisar
una fachada marcada como camión al 22% y motos duplicadas al 8–26%. Los pozos mantienen
su filtro independiente del 50%. `--proximity-conf 0.30` excluye esos vehículos débiles
de los avisos; el perfil se volvió a medir con ese filtro. La menor sensibilidad visual
puede ocultar objetos reales lejanos; no se afirma eliminar todos los falsos positivos.
Debug agrega corredor, depth map pequeño, profundidad de candidatos, persistencia, umbrales,
cajas raw de cebras y tiempo de los modelos. En output normal se omite esa información.

Analytics incluye `stage_performance`: tracking general, detector vial, analytics/evidencias,
scene, renderer, writer, decode y total por frame, con media y P95. `scene.crosswalk` y
`scene.depth` contienen tiempos por inferencia; `*_proximity.jsonl` conserva candidatos y estados
por frame. Un fallo de una capa opcional durante el procesamiento queda registrado y no
interrumpe el detector general ni los pozos; una ruta/modelo mal configurada falla al iniciar.

Los candidatos se separan en `dataset_candidates/potholes/{true_positive,false_positive,false_negative,unreviewed}`.
La recolección automática siempre guarda en `unreviewed`; una revisión visual explícita
puede organizar ejemplos con `scripts/collect_reviewed_candidates.py`. Sus etiquetas son
provisionales, sin entrenamiento y sin certificar pozos reales. Un score alto no convierte
una tapa, grieta o cordón en un pozo. El informe de la iteración queda en
`outputs/validation/v3_1/review.md`.
