"""Static Leaflet HTML: local event data/evidence, online OpenStreetMap background."""

import base64
import html
import io
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from .config import MIN_POTHOLE_CONFIDENCE
from .location import LocationProvider

ASSETS = Path(__file__).resolve().parents[1] / "assets/leaflet"


def evidence_preview(event: dict, events_path: Path, map_path: Path) -> str:
    reference = event.get("evidence_image") or event.get("evidence_path")
    if not reference:
        return '<p class="muted">Sin imagen de evidencia.</p>'
    path = (events_path.parent / reference).resolve()
    try:
        with Image.open(path) as image:
            detail = None
            box = event.get("bbox")
            if box and len(box) == 4 and all(math.isfinite(v) for v in box):
                x1,y1,x2,y2 = box
                padx,pady = max(60,(x2-x1)*.6),max(60,(y2-y1)*.6)
                bounds = max(0,int(x1-padx)),max(0,int(y1-pady)),min(image.width,int(x2+padx)),min(image.height,int(y2+pady))
                if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                    crop = image.crop(bounds)
                    crop.thumbnail((720,720))
                    buffer = io.BytesIO()
                    crop.convert("RGB").save(buffer,format="JPEG",quality=85)
                    detail = base64.b64encode(buffer.getvalue()).decode("ascii")
            image.thumbnail((720, 720))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=78)
        preview = base64.b64encode(buffer.getvalue()).decode("ascii")
        relative = quote(os.path.relpath(path, map_path.parent.resolve()).replace("\\", "/"), safe="/")
        link = html.escape(relative, quote=True)
        whole = (f'<a href="{link}" target="_blank" rel="noopener"><img class="evidence" '
                f'src="data:image/jpeg;base64,{preview}" alt="Evidencia del pozo"></a>'
                )
        photo = (f'<a class="photo-link" href="{link}" target="_blank" rel="noopener">Abrir foto original</a>')
        if detail:
            return (f'<img class="evidence detail" src="data:image/jpeg;base64,{detail}" '
                    'alt="Detalle del área detectada en la foto existente">'
                    f'<details><summary>Ver fotograma completo</summary>{whole}</details>{photo}')
        return whole + photo
    except (OSError, ValueError):
        return '<p class="muted">Imagen no disponible.</p>'


def event_popup(event: dict, events_path: Path, map_path: Path) -> str:
    source = {"mock": "Simulada (mock)", "windows": "Ubicación de Windows", "phone": "Ubicación del iPhone", "manual": "Ubicación marcada manualmente"}.get(event["location_source"], "GPS real")
    accuracy = event.get("location_accuracy_m")
    precision = f'<dt>Precisión</dt><dd>±{accuracy:.0f} m</dd>' if accuracy is not None else ''
    return (f'<h3>POZO · EVENTO #{int(event["event_id"])}</h3><dl>'
            f'<dt>Track</dt><dd>#{int(event["track_id"])}</dd>'
            f'<dt>Confianza</dt><dd>{event["confidence"]:.0%}</dd>'
            f'<dt>Tiempo</dt><dd>{event["timestamp"]:.3f} s</dd>'
            f'<dt>Fotograma</dt><dd>{int(event["frame"])}</dd>'
            f'<dt>Ubicación</dt><dd>{source}</dd>{precision}</dl>'
            + evidence_preview(event, events_path, map_path))


def render_map(provider: LocationProvider, events: list[dict], events_path: Path, output: Path,
               *, complete: bool, processed_seconds: float,
               confidence: float = MIN_POTHOLE_CONFIDENCE, enhanced: bool = True) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    valid = [e for e in events if e["confidence"] >= max(MIN_POTHOLE_CONFIDENCE, confidence)
             and e.get("latitude") is not None and e.get("longitude") is not None]
    title = {"mock": "RUTA SIMULADA — TRINIDAD", "windows": "RECORRIDO · UBICACIÓN DE WINDOWS", "phone": "RECORRIDO · UBICACIÓN DEL IPHONE", "manual": "RECORRIDO REVISADO · TRINIDAD"}.get(provider.source, "RECORRIDO CON GPS REAL")
    segments = getattr(provider, 'segments', (provider.points,))
    data = {"route": [[p.latitude, p.longitude] for p in provider.points],
            "segments": [[[p.latitude, p.longitude] for p in segment] for segment in segments],
            "events": [{"latitude": e["latitude"], "longitude": e["longitude"],
                        "id": e["event_id"], "confidence": e["confidence"], "timestamp": e["timestamp"],
                        "popup": event_popup(e, events_path, output)} for e in valid]}
    document = TEMPLATE.replace("__LEAFLET_CSS__", (ASSETS / "leaflet.css").read_text(encoding="utf-8"))
    document = document.replace("__LEAFLET_JS__", (ASSETS / "leaflet.js").read_text(encoding="utf-8"))
    document = document.replace("__TITLE__", title)
    document = document.replace("__COUNT__", str(len(valid)))
    average = sum(e['confidence'] for e in valid) / len(valid) if valid else None
    document = document.replace("__AVERAGE__", f"{average:.0%}" if average is not None else "—")
    document = document.replace("__SOURCE__", {"mock": "MOCK / SIMULADA", "windows": "WINDOWS", "phone": "IPHONE", "manual": "MARCADA MANUALMENTE"}.get(provider.source, "GPS REAL"))
    duration = max(0, processed_seconds)
    document = document.replace("__DURATION__", f"{duration:.1f} s")
    distance = ""
    if provider.source in {"real", "windows", "phone", "manual"}:
        meters = 0.0
        pairs = (pair for segment in segments for pair in zip(segment, segment[1:]))
        for a, b in pairs:
            lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
            dlat, dlon = lat2 - lat1, math.radians(b.longitude - a.longitude)
            hav = math.sin(dlat / 2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon / 2)**2
            meters += 6371000 * 2 * math.asin(math.sqrt(min(1, hav)))
        label = ("Distancia dibujada · aproximada" if provider.source == "manual" else
                 "Recorrido GPS estimado" if provider.source == "real" else "Distancia registrada · estimada")
        distance = f"<span>{label}</span><strong>{meters / 1000:.2f} km</strong>"
    document = document.replace("__DISTANCE__", distance)
    partial = 'Recorrido parcial' if provider.source in {'windows', 'phone'} else 'Prueba parcial'
    document = document.replace("__STATUS__", "Recorrido completo" if complete else f"{partial} · {processed_seconds:.1f} s procesados")
    document = document.replace("__DISCLAIMER__", "Simulación para desarrollo. No representa las ubicaciones reales del video."
                                if provider.source == "mock" else
                                "Recorrido y posiciones de pozos marcados manualmente al revisar las evidencias."
                                if provider.source == "manual" else "Ubicación aproximada del vehículo al detectar el daño.")
    document = document.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"))
    if not enhanced:
        document = document.replace('</style></head>', '.scan,.metrics{display:none}.evidence{height:260px}</style></head>')
    output.write_text(document, encoding="utf-8")
    return {"map_path": str(output.resolve()), "geolocated_potholes": len(valid),
            "average_pothole_confidence": average}


TEMPLATE = r'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer-when-downgrade">
<title>Urban Vision · Estado de la calle</title><style>__LEAFLET_CSS__</style>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;color:#eaf1ed;background:#171e22}
#map{height:100vh;width:100vw;background:#283238}.summary{position:absolute;z-index:650;top:24px;left:60px;width:350px;padding:24px;background:rgba(20,29,32,.95);border-top:3px solid #8be6ce;border-radius:4px;box-shadow:0 8px 30px #0003}
h1{font-size:24px;font-weight:500;letter-spacing:2px;margin:0 0 14px}.badge{font-size:12px;font-weight:bold;letter-spacing:1px;color:#ecc17b}.description,.status{font-size:12px;line-height:1.5;color:#acb9bc}
.count{font-size:36px;color:#ecc17b;margin:16px 0 0}.count span{font-size:13px;color:#eaf1ed;margin-left:12px}.legend{display:flex;gap:18px;font-size:12px;margin:18px 0}.line{color:#8be6ce}.point{color:#ecc17b}
#events{max-height:180px;overflow:auto;border-top:1px solid #506064;padding-top:8px}.event{display:block;width:100%;border:0;border-bottom:1px solid #354347;padding:10px 0;background:none;color:#eaf1ed;text-align:left;cursor:pointer;font-size:12px}.event:hover{color:#8be6ce}
.scan{color:#8be6ce;font-size:11px;letter-spacing:2px;margin-bottom:12px}.metrics{display:grid;grid-template-columns:1fr auto;gap:7px;font-size:12px;color:#acb9bc;margin:14px 0}.metrics strong{color:#eaf1ed;font-weight:normal}.route-end{background:#192428;border:1px solid #8be6ce;border-radius:3px;color:#eaf1ed;padding:3px 6px;font:10px Arial;white-space:nowrap}
.leaflet-popup-content{font-family:Arial,sans-serif;width:320px!important;max-width:70vw}h3{font-size:14px;letter-spacing:.5px;margin:0 0 12px}dl{display:grid;grid-template-columns:95px 1fr;font-size:12px;gap:5px}dt{color:#637277}dd{margin:0}.evidence{display:block;height:360px;max-height:40vh;width:auto;max-width:100%;margin:12px auto;border-radius:3px}.photo-link{display:block;text-align:center;font-size:12px}.muted{color:#637277;font-size:12px}
.evidence.detail{width:100%;height:auto;object-fit:contain;max-height:280px}details{font-size:12px;margin:12px 0}summary{cursor:pointer;color:#637277}
#tile-warning{display:none;position:absolute;bottom:30px;left:20px;z-index:900;background:#192428;color:#ecc17b;padding:10px;font-size:12px;border-radius:3px}
@media(max-width:600px){.summary{top:12px;left:48px;width:calc(100vw - 64px);padding:16px}h1{font-size:20px}.description{margin:8px 0}#events{max-height:100px}.count{font-size:28px;margin-top:8px}.legend{margin:10px 0}}
@media(max-width:600px){body.popup-open .summary{padding:10px 16px}body.popup-open .summary .description,body.popup-open .summary .status,body.popup-open .summary .metrics,body.popup-open .summary .legend,body.popup-open #events,body.popup-open .scan{display:none}body.popup-open h1{font-size:16px;margin-bottom:5px}body.popup-open .count{font-size:20px;margin-top:5px}.leaflet-popup-content{width:280px!important;max-width:76vw}}
</style></head><body><div id="map" aria-label="Mapa del recorrido y pozos detectados"></div>
<aside class="summary"><h1>URBAN VISION</h1><div class="scan">ANÁLISIS URBANO</div><div class="badge">__TITLE__</div><p class="description">__DISCLAIMER__</p>
<div class="count">__COUNT__<span>POZOS REGISTRADOS · EST.</span></div><p class="status">__STATUS__</p>
<div class="metrics"><span>Duración procesada</span><strong>__DURATION__</strong><span>Confianza promedio</span><strong>__AVERAGE__</strong><span>Ubicación</span><strong>__SOURCE__</strong>__DISTANCE__</div>
<div class="legend"><span class="line">━ Recorrido</span><span class="point">● Pozos</span></div><div id="events"></div></aside>
<div id="tile-warning" role="status">Fondo de calles no disponible. El recorrido y las evidencias siguen accesibles.</div>
<script>__LEAFLET_JS__</script><script>
const data=__DATA__;
const map=L.map('map',{zoomControl:false});
L.control.zoom({zoomInTitle:'Acercar',zoomOutTitle:'Alejar'}).addTo(map);
map.on('popupopen',event=>{document.body.classList.add('popup-open');Object.assign(event.popup.options,popupLayout());event.popup.update();const close=event.popup.getElement().querySelector('.leaflet-popup-close-button');if(close){close.title='Cerrar';close.setAttribute('aria-label','Cerrar popup');}});
map.on('popupclose',()=>document.body.classList.remove('popup-open'));
// A file:// page cannot supply the web Referer required by OSM's tile policy.
// Use the real local HTTP origin; never forge headers or proxy blocked requests.
const warning=document.getElementById('tile-warning');
if(location.protocol==='http:' || location.protocol==='https:'){
 const tiles=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,referrerPolicy:'no-referrer-when-downgrade',attribution:'&copy; colaboradores de <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'});
 tiles.on('tileerror',()=>warning.style.display='block');tiles.addTo(map);
}else{
 warning.textContent='Para cargar las calles, abrí este mapa con el visor local de Urban Vision.';
 warning.style.display='block';
 map.attributionControl.addAttribution('&copy; colaboradores de <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>');
}
const route=L.polyline(data.segments,{color:'#42bca6',weight:4,opacity:.85}).addTo(map);
if(data.route.length){
 map.fitBounds(route.getBounds(),{padding:[30,30],maxZoom:17});
 [[data.route[0],'INICIO'],[data.route[data.route.length-1],'FIN']].forEach(([point,label])=>L.circleMarker(point,{radius:4,color:'#42bca6',fillColor:'#192428',fillOpacity:1,weight:2}).addTo(map).bindTooltip(label,{permanent:true,direction:label==='INICIO'?'top':'bottom',className:'route-end'}));
}else{map.setView([0,0],2);document.querySelector('.status').textContent+=' · Sin posiciones registradas';}
const markers=[];const list=document.getElementById('events');
function popupLayout(){
 const panel=document.querySelector('.summary').getBoundingClientRect();
 const beside=innerWidth-panel.right>390;
 return {maxWidth:360,maxHeight:Math.max(120,beside?innerHeight-100:innerHeight-panel.bottom-70),autoPanPaddingTopLeft:beside?[panel.right+20,20]:[20,panel.bottom+20],autoPanPaddingBottomRight:[20,20]};
}
data.events.forEach(e=>{const marker=L.circleMarker([e.latitude,e.longitude],{radius:7,color:'#654b25',weight:1.5,fillColor:'#e8b56b',fillOpacity:e.confidence>=.7?1:.55}).addTo(map).bindPopup(e.popup,popupLayout());markers.push(marker);
const button=document.createElement('button');button.className='event';button.textContent=`POZO #${e.id} · ${(e.confidence*100).toFixed(0)}% · ${e.timestamp.toFixed(2)} s`;
button.addEventListener('click',()=>{map.panTo(marker.getLatLng(),{animate:false});marker.openPopup();});list.appendChild(button);});
if(!data.events.length){list.textContent='Sin pozos confirmados con el umbral configurado.';list.className='description';}
</script></body></html>'''
