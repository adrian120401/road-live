"""Assemble measured outputs and explicit visual-review notes for this iteration."""

from collections import Counter
import json
from pathlib import Path


def main() -> None:
    root = Path("outputs/validation/v3_1")
    report = json.loads(Path("outputs/urban_vision_v3_1_output.json").read_text(encoding="utf-8"))
    baseline = json.loads((root/"baseline.json").read_text(encoding="utf-8"))
    comparison = json.loads((root/"comparison.json").read_text())
    scene,road = report["scene"],report["road_damage"]
    proximity = scene["proximity"]
    trace = [json.loads(line) for line in Path(proximity["trace_path"]).read_text().splitlines()]
    controls = [1,180,600,900,1550,1650,1800]
    visual = {"reviewed_control_frames":controls,"control_states":{str(n):trace[n-1]["state"] for n in controls},
              "reviewed_warning_frames":[1048,1133,1191,1375,1393,1426,1459],
              "warning_targets":"Forward motorcycle/rider, with pickup ahead; no lateral parked vehicle in reviewed warning frames",
              "confirmed_lateral_false_warnings_in_reviewed_frames":0,
              "limitations":"Sampled visual inspection, not ground truth for every frame; candidate ID switches remain."}
    (root/"proximity_visual_review.json").write_text(json.dumps(visual,indent=2))
    warning_ids = {state:sorted({t["track_id"] for t in proximity["transitions"] if t["state"]==state})
                   for state in ("CAUTION","NEAR")}
    thresholds = proximity["thresholds"]
    stats = report["stage_performance"]
    rows = "\n".join(f"| {name} | {stats[name]['mean_ms']:.2f} | {stats[name]['p95_ms']:.2f} |"
                     for name in ("general_tracking","road_damage","road_analytics_evidence","analytics",
                                  "scene","renderer","writer","decode","total_frame"))
    transitions = "\n".join(f"| {t['frame']} | {t['timestamp']:.2f} | {t['track_id']} | {t['state']} | {t['depth_estimate']:.2f} | {t['persistent_frames']} |"
                            for t in proximity["transitions"])
    text = f"""# Urban Vision V3.1 — revisión del video2.MOV

## Resultado y continuidad

Video completo: {report['frames_processed']} frames, 2160×3840, {report['source_fps']:.5f} FPS de reproducción,
63,829 s. GTX 1050 Ti / torch 2.7.1+cu118 / Ultralytics 8.4.171. Sin entrenamiento ni inferencia cloud.
General YOLO26n + BoT-SORT; arquitectura anterior extendida con capas opcionales.
CLI separado en `src/cli.py`; `scene`, `crosswalk`, `depth`, `proximity`, `performance`
aislan la nueva lógica. Road association conserva su algoritmo; geometría compartida en `association.py`.

Referencia V3 completa: **{baseline['processing_fps']:.3f} FPS**. Final V3.1: **{report['processing_fps']:.3f} FPS**,
variación **{comparison['fps_change_percent']:+.2f}%**. Pasadas secuenciales, sin un benchmark estadístico de carga externa.
Mismos **235 IDs**, frames de primera/última aparición, observaciones, continuidad y gaps.
Mismos **14 eventos viales**, coordenadas/timestamps/scores y **14 fotos idénticas por SHA256**.

## Mapa

14 marcadores con confianza >=50%, confianza promedio **{report['location']['average_pothole_confidence']:.1%}**,
duración procesada, origen y extremos INICIO/FIN. No se muestra distancia para mock.
ExifTool encontró una coordenada única (-33.5147, -56.8963) y fecha, sin trayectoria GPS temporal.
Se usó ruta **MOCK**, 65 puntos temporales en Trinidad: no representa el recorrido real.
No se generan GeoJSON nuevos; puntos en `location.trajectory` del JSON de analytics.

Popups: detalle del área detectada a partir de la misma foto, fotograma desplegable y original 4K.
No se crean fotos adicionales para el mapa. HTML de aproximadamente 1,48 MB con previews incrustadas.
Navegador: 14 enlaces HTTP 200, fotos visibles en escritorio/móvil, sin excepciones JS.
OpenStreetMap responde HTTP 200 usando origen HTTP real y cache normal, sin falsificar headers.
Los datos y fotos incrustadas funcionan offline; las calles requieren conexión y el visor local.
Capturas/auditoría: `output/playwright/v31-final*`.

## Detección general y semáforos

Clases verificadas en el checkpoint: person, car, motorcycle, bicycle, bus, truck, traffic light, stop sign.
Conteos finales aproximados por ID: `{json.dumps(report['counts'])}`.
Semáforos: **8 IDs**, con cajas visibles en la intersección (p.ej. frame1191).
No son necesariamente ocho semáforos físicos: las pérdidas de asociación pueden fragmentar IDs.
No se determina rojo/amarillo/verde. Hay omisiones de semáforos pequeños/lejanos (p.ej. frame1049).

Categoría por voto ponderado en diez observaciones y tres mayorías sostenidas; corregir una
clase transfiere su cuenta, no crea un ID. Corrige la etiqueta MOTO congelada sobre el conductor
(ID1056 acaba como PERSONA). Algunas confusiones persona/moto y cajas parciales siguen presentes.
La caja de una fachada fue inicialmente ÓMNIBUS y luego CAMIÓN al 22%; el gate visual la oculta.
Se ocultan cajas generales <30%, preservando los candidatos originales del tracker al 10%,
analytics y las clases pequeñas de infraestructura. Gates de pozos y PARE permanecen independientes.
Puede ocultar objetos reales de baja confianza; no se elevó arbitrariamente la confianza de tracking.

## Cebras

COCO no tiene esta clase. Especializado revisado: kairess/crosswalk-traffic-light-detection-yolov5,
revisión `5ba1e01d946e1a5cc9dcf396b3b8fc62ae2666d5`, SelectStar declarado, YOLOv5s6.
Conversión ONNX con seis verificaciones de paridad; cajas incorrectas en lateral del cruce y
~316 ms CPU/inferencia. Se conservan exportador, SHA y manifest, pero no se usa en la demo final.

Elegido: **YOLOE-26s-seg oficial**, visual prompting desde frame270, bbox(12,2180,2155,2510).
Los prompts de texto y el recorte de pavimento antes de inferir no confirmaban el cruce.
Un ejemplo visual fija embeddings locales; no hubo entrenamiento ni text encoder durante inferencia.
Se mantiene contexto completo y se filtra por ROI0/.42/1/.74, al menos 80% de caja dentro,
confidence.35, NMS.40 (duplicados revisados en frames271/274), intervalo3, confirmación de tres inferencias.
**1 cruce**, track1: primera propuesta259, confirmación271, último286, máximo74.4%.
Se inspeccionó la caja sobre las franjas en frames271/277/286 y controles de otras escenas sin cebra.
En frame1550 hay marcas blancas desgastadas compatibles con otro paso: no se confirmó.
Esto deja un candidato de falso negativo de cebras para revisión, además del cruce claro detectado.
**Validación en la misma escena de referencia; no es evidencia de generalización a otras calles.**
Se usa caja discreta, no máscara sólida de media pantalla.

Fuentes: [autor especializado](https://github.com/kairess/crosswalk-traffic-light-detection-yolov5),
[YOLOE oficial](https://docs.ultralytics.com/models/yoloe/).

## Pozos y candidatos de dataset

Mismo modelo PeterHdd/YOLOv8s de V2. Confidence obligatorio **0.50**, intervalo2,
ROI(.05,.48,.95,.70), confirmación2, NMS.45. {road['potholes']} eventos válidos por score,
{road['saved_evidence']} fotos geolocalizadas. **{road['discarded_below_50_observations']} observaciones** en [35%,50%) descartadas,
no 80 pozos distintos; scores menores a35% no se miden. Nada <50% entra en HUD, eventos,
fotos, candidatos automáticos o mapa. Un score alto no certifica una cavidad real.

Revisión visual provisional (`pothole_review.json`): **10 falsos positivos de categoría**
(rejilla/tapa, cordón, grietas/juntas), **2 fotos compatibles con un pozo** que probablemente
corresponden a **una misma zona dañada**, y **2 ambiguos**. Necesita revisión humana antes de etiquetar
un dataset definitivo. Eventos11/12 muestran fragmentación temporal; el conteo no se considera perfecto.
No se encontraron pozos completamente omitidos que puedan certificarse desde estas muestras.
Se guardó frame1700 como **candidato de detección tardía**: daño lejano aparentemente detectado
después; no equivale a un pozo físico perdido por todo el video. Sin entrenamiento.

`dataset_candidates/potholes/`: true_positive(2), false_positive(10), false_negative(1 candidato),
unreviewed(2 ambiguos + las14 recolecciones automáticas). Se conserva procedencia y revisión en JSON.
El colector no decide labels a partir del score; aplica el manifiesto explícito de revisión.

## Proximidad experimental

Modelo oficial **YOLO26n-depth**, lado máximo768, intervalo3 y solo con vehículo candidato;
{scene['depth']['performance']['inferences']} inferencias de profundidad (456 antes del filtro de candidatos).
Autos/motos/buses/trucks con confidence>=30%, centro y base dentro del trapecio medido.
Mediana central/inferior de caja, rechazo de inválidos y P10/P90, al menos16 píxeles válidos y75% del área.
Muestras cacheadas solo con mismo ID, caja compatible y vigencia limitada; cinco valores temporales.
Cinco frames de candidato, dos muestras nuevas para subir y tres para bajar; cambio/pérdida de ID reinicia.

Perfil específico de esta cámara: `assets/proximity_video2.json`.
Median SAFE={thresholds['groups']['SAFE']['median']:.2f}, CAUTION={thresholds['groups']['CAUTION']['median']:.2f},
NEAR={thresholds['groups']['NEAR']['median']:.2f}; grupos revisados950–1000/1050–1110/1170–1250.
Umbrales: **NEAR<={thresholds['near_threshold']:.2f}**, **CAUTION<={thresholds['caution_threshold']:.2f}**,
histéresis **{thresholds['hysteresis']:.2f}**, en unidades nominales del modelo, **no metros calibrados ni distancias de seguridad**.
Medias geométricas entre medianas y P90 de variación temporal; no hay umbrales inventados para otra cámara.
Sin perfil, solo diagnóstico de profundidad. Resolución/corredor/gate incompatibles con perfil fallan al iniciar.

CAUTION IDs `{warning_ids['CAUTION']}`; NEAR IDs `{warning_ids['NEAR']}`.
Corresponden al vehículo/motociclista delantero en escenas revisadas. No hubo avisos laterales falsos
en los controles revisados; no se afirma validar cada objeto de todos los frames.
Persisten cambios de candidato entre IDs/cajas parciales del mismo motociclista, que pueden provocar
breves desapariciones/reapariciones del panel. No se modificó el tracker general para ocultar este límite.
El corredor fijo no determina carril real, orientación, estado estacionado ni movimiento de aproximación.
Estado no indica peligro de colisión y también puede aparecer mientras ambos vehículos están detenidos.

| Frame | Tiempo(s) | ID | Estado | Depth nominal | Persistencia(frames) |
|---:|---:|---:|---|---:|---:|
{transitions}

Fuente: [profundidad oficial](https://docs.ultralytics.com/tasks/depth/).

## Rendimiento

Por inferencia, warmup excluido: general **{report['general_performance']['mean_inference_ms']:.2f} ms**;
road **{road['performance']['mean_inference_ms']:.2f} ms**;
crosswalk **{scene['crosswalk']['performance']['mean_inference_ms']:.2f} ms**;
depth **{scene['depth']['performance']['mean_inference_ms']:.2f} ms**.
General cadaframe; road cada2; crosswalk cada3; depth cada3 únicamente con candidato.
Trail overlay ahora copia/mezcla solo la región de sus puntos, manteniendo resolución y apariencia.

| Etapa completa por frame (incluye frames sin inferencia) | Media(ms) | P95(ms) |
|---|---:|---:|
{rows}

Cuello de botella principal: tracking/preprocesamiento general, lectura y exportación4K en CPU;
la inferencia GPU aislada no explica el FPS total. Mantener native2160×3840 cuesta más que trabajar
sobre el antiguo video pequeño. El output está sin audio y recodificado MP4v, como V1/V2.

## Validación y archivos

79 tests CPU/sin descargas; compilación Python; smoke360frames y exportación completa.
Auditoría de1915frames, FPS/resolución, conteos por ID, intervals, evidencias/PTS/geolocalización,
corredor/persistencia de avisos, ausencia de errores de capas y popups en navegador.
No hubo warnings de inferencia en la exportación final. La preparación legacy sí mostró un
TracerWarning esperado por export fixedshape y problemas de compatibilidad/dependencias ya resueltos;
el export quedó verificado, sin incorporarlo al runtime final.

- Video: `outputs/urban_vision_v3_1_output.mp4`
- Analytics/perfil: `outputs/urban_vision_v3_1_output.json`
- Eventos: `outputs/urban_vision_v3_1_output_events.json`
- Evidencias: `outputs/urban_vision_v3_1_output_events/`
- Mapa: `outputs/map.html`
- Candidatos/estado: `outputs/urban_vision_v3_1_output_proximity.jsonl`
- Auditoría/frames: `outputs/validation/urban_vision_v3_1_output/`
- Comparación: `outputs/validation/v3_1/comparison.json`

Abrir calles del mapa: `python scripts/open_map.py --map outputs/map.html`.

## Antes del recorrido definitivo

1. Revisar y etiquetar negativos locales de tapas/grietas/cordones y deduplicar el daño11/12;
   luego evaluar fine-tuning pequeño, sin entrenar con labels automáticos sin revisar.
2. Validar el perfil visual de cebras con otros cruces y negativos de pavimento; esta única escena
   no valida generalización. Sustituir por modelo especializado local si se consigue dataset/pesos fiables.
3. Agrupar cajas parciales y reducir cambios del candidato de proximidad, manteniendo IDs originales;
   revisar semáforos pequeños con una variante mayor/resolución de inferencia y medir impacto.
4. Montaje firme, menos cielo/tablero y más pavimento; conservar archivo original, luz uniforme,
   varios segundos antes/durante/después de daños claros y vehículos delanteros/laterales.
   Cambiar ángulo exige nueva ROI/corredor/perfil.
5. Para ubicación real, verificar muestras GPS temporales sincronizadas antes del recorrido;
   una única ubicación del MOV no basta. Mock sigue claramente indicado hasta tener ese track.
"""
    (root/"review.md").write_text(text,encoding="utf-8")
    print("Review written:",root/"review.md",flush=True)


if __name__ == "__main__":
    main()
