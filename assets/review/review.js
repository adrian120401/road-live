'use strict';

const $ = id => document.getElementById(id);
const state = {project: null, mode: 'route', selected: null, vertex: null, placing: false,
  undo: [], generation: 0, savedGeneration: 0, saving: false, saveError: false, timer: null,
  photoFull: false, exportRunning: false};
const map = L.map('map', {zoomControl: false}).setView([-33.5147, -56.8984], 16);
L.control.zoom({position: 'bottomright', zoomInTitle: 'Acercar', zoomOutTitle: 'Alejar'}).addTo(map);
const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, referrerPolicy: 'no-referrer-when-downgrade',
  attribution: '&copy; colaboradores de <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);
tiles.on('tileerror', () => {$('tile-warning').hidden = false;});
const routeLayer = L.layerGroup().addTo(map);
const eventLayer = L.layerGroup().addTo(map);
const line = L.polyline([], {color: '#42bca6', weight: 4, opacity: .95}).addTo(map);

function timeLabel(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, '0')}:${(seconds % 60).toFixed(2).padStart(5, '0')}`;
}
function error(message) { $('error-banner').textContent = message; $('error-banner').hidden = !message; }
function draftKey() {return `urban-vision-review:${state.project.source.sha256}`;}
function progress() {
  const events = state.project.events;
  const accepted = events.filter(e => e.status === 'accepted' && e.position).length;
  const rejected = events.filter(e => e.status === 'rejected').length;
  const pending = events.length - accepted - rejected;
  const routeReady = state.project.route.length >= 2 && new Set(state.project.route.map(p => p.join(','))).size >= 2;
  return {accepted, rejected, pending, ready: routeReady && pending === 0 && state.project.analysis.complete};
}
function selectedEvent() {return state.project?.events.find(e => e.event_id === state.selected);}
function touch() {
  state.generation++;
  state.saveError = false;
  $('save-state').textContent = 'Cambios sin guardar';
  $('save-state').dataset.state = '';
  try {localStorage.setItem(draftKey(), JSON.stringify(state.project));} catch (_) { /* Server save remains authoritative. */ }
  clearTimeout(state.timer);
  state.timer = setTimeout(save, 450);
  renderSummary();
}
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'No se pudo completar la operación.');
  return data;
}
async function save() {
  clearTimeout(state.timer);
  if (state.saving) return;
  state.saving = true;
  $('retry-save').hidden = true;
  try {
    while (state.savedGeneration !== state.generation) {
      const generation = state.generation;
      const payload = structuredClone(state.project);
      $('save-state').textContent = 'Guardando…';
      const data = await api('/api/project', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      state.project.revision = data.project.revision;
      state.project.updated_at = data.project.updated_at;
      state.savedGeneration = generation;
      try {
        if (generation === state.generation) localStorage.removeItem(draftKey());
        else localStorage.setItem(draftKey(), JSON.stringify(state.project));
      } catch (_) {}
    }
    state.saveError = false;
    $('save-state').textContent = 'Guardado en tu equipo';
    $('save-state').dataset.state = 'saved';
    error('');
  } catch (err) {
    state.saveError = true;
    $('save-state').textContent = 'No se pudo guardar';
    $('save-state').dataset.state = 'error';
    $('retry-save').hidden = false;
    error(`${err.message} Los cambios siguen en esta pestaña y en el borrador local.`);
  } finally {
    state.saving = false;
    renderSummary();
  }
}
function renderSummary() {
  const p = progress();
  $('review-count').textContent = p.accepted;
  $('review-summary').textContent = `${p.pending} pendientes · ${p.accepted} ubicados · ${p.rejected} descartados`;
  $('export').disabled = !p.ready || state.saveError || state.saving || state.savedGeneration !== state.generation || state.exportRunning;
  $('export').title = p.ready ? 'Crear el video con los pozos revisados y el mapa al cierre' : 'Dibujá el recorrido y ubicá o descartá todas las detecciones';
  const route = state.project.route;
  $('route-summary').textContent = route.length ? `${route.length} puntos · ${route.length < 2 ? 'marcá el siguiente punto' : 'inicio a fin'}` : 'Sin recorrido';
  let distance = 0;
  route.slice(1).forEach((point, i) => {distance += L.latLng(route[i]).distanceTo(L.latLng(point));});
  $('route-distance').textContent = route.length > 1 ? `${(distance / 1000).toFixed(2)} km dibujados` : '';
  $('undo').disabled = !state.undo.length || state.mode !== 'route';
  $('delete-point').disabled = state.vertex === null || state.mode !== 'route';
}
function markerIcon(text, className, selected) {
  return L.divIcon({className: `${className}${selected ? ' selected' : ''}`, html: String(text), iconSize: [26, 26], iconAnchor: [13, 13]});
}
function accessibleMarker(marker, label, onSelect) {
  const element = marker.getElement();
  element.setAttribute('aria-label', label);
  element.setAttribute('role', 'button');
  element.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); onSelect();}
  });
}
function renderRoute() {
  routeLayer.clearLayers();
  const route = state.project.route;
  line.setLatLngs(route);
  route.forEach((point, index) => {
    const label = index === 0 ? 'Inicio' : index === route.length - 1 ? 'Fin' : `Giro ${index}`;
    const marker = L.marker(point, {icon: markerIcon(index + 1, 'route-vertex', state.vertex === index),
      draggable: state.mode === 'route', keyboard: true, bubblingMouseEvents: false}).addTo(routeLayer);
    marker.bindTooltip(label, {direction: 'top'});
    const select = () => {state.vertex = index; renderRoute(); renderSummary();};
    marker.on('click', select);
    marker.on('dragstart', () => {state.undo.push(structuredClone(route));});
    marker.on('dragend', () => {
      const position = marker.getLatLng();
      route[index] = [position.lat, position.lng];
      state.vertex = index;
      touch(); renderRoute();
    });
    accessibleMarker(marker, `${label}, punto ${index + 1}. Arrastrá para mover.`, select);
  });
  renderSummary();
}
function renderMarkers() {
  eventLayer.clearLayers();
  state.project.events.filter(e => e.status === 'accepted' && e.position).forEach(event => {
    const marker = L.marker(event.position, {icon: markerIcon(event.event_id, 'pozo-marker', event.event_id === state.selected),
      draggable: state.mode === 'potholes', keyboard: true, bubblingMouseEvents: false}).addTo(eventLayer);
    marker.bindTooltip(`Pozo #${event.event_id}`, {direction: 'top'});
    const select = () => selectEvent(event.event_id);
    marker.on('click', select);
    marker.on('dragend', () => {
      const position = marker.getLatLng();
      event.position = [position.lat, position.lng];
      touch(); renderEvidence();
    });
    accessibleMarker(marker, `Pozo ${event.event_id}. Seleccionar evidencia.`, select);
  });
}
function setMode(mode) {
  state.mode = mode;
  state.placing = false;
  $('route-mode').classList.toggle('selected', mode === 'route');
  $('pothole-mode').classList.toggle('selected', mode === 'potholes');
  $('route-mode').setAttribute('aria-pressed', String(mode === 'route'));
  $('pothole-mode').setAttribute('aria-pressed', String(mode === 'potholes'));
  $('map-hint').textContent = mode === 'route'
    ? 'Marcá el inicio y después cada esquina o giro. Arrastrá los puntos para corregirlos.'
    : 'Elegí una foto, ubicá el pozo en el mapa o descartá la detección.';
  renderRoute(); renderMarkers();
}
function renderEvents() {
  const container = $('events');
  container.replaceChildren();
  const filter = $('event-filter').value;
  const visible = state.project.events.filter(e => filter === 'all' || e.status === filter);
  visible.forEach(event => {
    const observation = event.observation;
    const button = document.createElement('button');
    button.className = `event-row${event.event_id === state.selected ? ' selected' : ''}`;
    button.dataset.status = event.status;
    button.setAttribute('aria-pressed', String(event.event_id === state.selected));
    const title = document.createElement('span');
    title.textContent = `Pozo #${event.event_id}`;
    const detail = document.createElement('small');
    detail.textContent = `${timeLabel(observation.timestamp)} · ${(observation.confidence * 100).toFixed(0)}% de confianza`;
    title.append(detail);
    const status = document.createElement('span');
    status.className = 'event-status';
    status.textContent = {pending: 'Pendiente', accepted: 'Ubicado', rejected: 'Descartado'}[event.status];
    button.append(title, status);
    button.addEventListener('click', () => selectEvent(event.event_id));
    container.append(button);
  });
  $('empty-state').hidden = visible.length > 0;
  $('empty-state').textContent = state.project.events.length
    ? 'No hay detecciones con este filtro.'
    : 'No se confirmaron detecciones de pozos. Podés dibujar el recorrido y exportar el mapa sin marcadores.';
  renderSummary();
}
function photoPath(observation, detail) {
  const path = detail ? observation.detail_path : observation.evidence_path;
  return path ? '/files/' + path.split('/').map(encodeURIComponent).join('/') : '';
}
function renderEvidence() {
  const event = selectedEvent();
  $('evidence-panel').hidden = !event;
  if (!event) return;
  const observation = event.observation;
  $('evidence-title').textContent = `Pozo #${event.event_id}`;
  $('evidence-time').textContent = timeLabel(observation.timestamp);
  $('evidence-meta').textContent = `${(observation.confidence * 100).toFixed(0)}% de confianza · fotograma ${observation.frame}`;
  const source = photoPath(observation, !state.photoFull) || photoPath(observation, false);
  $('photo-error').hidden = !!source;
  $('evidence-image').hidden = !source;
  if ($('evidence-image').getAttribute('src') !== source) $('evidence-image').src = source;
  $('photo-link').href = photoPath(observation, false);
  $('photo-detail').textContent = state.photoFull ? 'Ver detalle del pozo' : 'Ver foto completa';
  $('photo-detail').hidden = !observation.detail_path;
  $('reject').textContent = event.status === 'rejected' ? 'Descartado' : 'Descartar';
  $('reject').disabled = event.status === 'rejected';
  $('place').textContent = event.status === 'accepted' ? 'Mover en el mapa' : 'Ubicar en el mapa';
  $('reset-event').hidden = event.status === 'pending';
  $('position-label').textContent = event.position
    ? `Ubicación manual: ${event.position[0].toFixed(6)}, ${event.position[1].toFixed(6)}. Podés arrastrar el marcador.`
    : event.status === 'rejected' ? 'Esta detección no aparecerá en el video ni en el mapa final.' : 'Ubicación pendiente.';
}
function selectEvent(id) {
  state.selected = id;
  state.photoFull = false;
  state.placing = false;
  if (state.mode !== 'potholes') setMode('potholes');
  renderEvents(); renderEvidence(); renderMarkers();
  const event = selectedEvent();
  if (event.position) map.panTo(event.position);
  if ($('video-details').open) seekVideo();
}
function seekVideo() {
  const event = selectedEvent();
  if (!event) return;
  const video = $('preview');
  const seek = () => {video.currentTime = Math.max(0, event.observation.timestamp - 1.5); video.pause();};
  if (!video.getAttribute('src')) {
    video.addEventListener('loadedmetadata', seek, {once: true});
    video.src = '/files/' + state.project.analysis.preview;
    video.load();
  } else if (video.readyState >= 1) seek();
}
function placeMode() {
  if (!selectedEvent()) return;
  if (state.mode !== 'potholes') setMode('potholes');
  state.placing = true;
  $('map-hint').textContent = `Marcá sobre el mapa dónde estaba el pozo #${state.selected}. Escape cancela.`;
  if (innerWidth <= 720) $('map').scrollIntoView({block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
}
map.on('click', event => {
  if (!state.project) return;
  const point = [event.latlng.lat, event.latlng.lng];
  if (state.mode === 'route') {
    state.undo.push(structuredClone(state.project.route));
    state.project.route.push(point);
    state.vertex = state.project.route.length - 1;
    touch(); renderRoute();
  } else if (state.placing && selectedEvent()) {
    Object.assign(selectedEvent(), {position: point, status: 'accepted'});
    state.placing = false;
    $('map-hint').textContent = `Pozo #${state.selected} ubicado. Arrastrá su marcador para corregirlo.`;
    touch(); renderMarkers(); renderEvents(); renderEvidence();
  }
});
$('route-mode').addEventListener('click', () => setMode('route'));
$('pothole-mode').addEventListener('click', () => setMode('potholes'));
$('fit-route').addEventListener('click', () => {
  if (state.project.route.length > 1) map.fitBounds(line.getBounds(), {padding: [45, 65], maxZoom: 18});
  else map.setView(state.project.center, 16);
});
$('undo').addEventListener('click', () => {
  if (!state.undo.length) return;
  state.project.route = state.undo.pop(); state.vertex = null; touch(); renderRoute();
});
$('delete-point').addEventListener('click', () => {
  if (state.vertex === null) return;
  state.undo.push(structuredClone(state.project.route));
  state.project.route.splice(state.vertex, 1); state.vertex = null; touch(); renderRoute();
});
$('event-filter').addEventListener('change', renderEvents);
$('place').addEventListener('click', placeMode);
$('reject').addEventListener('click', () => {
  if (!selectedEvent()) return;
  Object.assign(selectedEvent(), {status: 'rejected', position: null}); state.placing = false;
  touch(); renderMarkers(); renderEvents(); renderEvidence();
});
$('reset-event').addEventListener('click', () => {
  Object.assign(selectedEvent(), {status: 'pending', position: null});
  touch(); renderMarkers(); renderEvents(); renderEvidence();
});
$('photo-detail').addEventListener('click', () => {state.photoFull = !state.photoFull; renderEvidence();});
$('evidence-image').addEventListener('error', () => {$('photo-error').hidden = false; $('evidence-image').hidden = true;});
$('video-details').addEventListener('toggle', () => {if ($('video-details').open) seekVideo(); else $('preview').pause();});
$('retry-save').addEventListener('click', save);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && state.placing) {state.placing = false; $('map-hint').textContent = 'Elegí una foto y luego ubicá el pozo en el mapa.';}
});
window.addEventListener('beforeunload', event => {
  if (state.generation !== state.savedGeneration) {event.preventDefault(); event.returnValue = '';}
});
async function pollExport() {
  try {
    const status = await api('/api/export');
    state.exportRunning = status.state === 'running';
    $('export-status').textContent = status.message;
    $('export-progress').hidden = status.state !== 'running';
    $('export-progress').value = status.progress || 0;
    if (status.state === 'done') {
      $('downloads').replaceChildren();
      [['video', 'Descargar video 4K'], ['map', 'Abrir mapa interactivo']].forEach(([key, text]) => {
        const link = document.createElement('a'); link.textContent = text; link.href = '/files/' + status[key];
        if (key === 'video') link.download = status[key]; else {link.target = '_blank'; link.rel = 'noopener';}
        $('downloads').append(link);
      });
    }
    renderSummary();
    if (state.exportRunning) setTimeout(pollExport, 1800);
  } catch (err) {
    $('export-status').textContent = 'No se pudo consultar la exportación. Revisá que el servidor siga abierto.';
    if (state.exportRunning) setTimeout(pollExport, 4000);
  }
}
$('export').addEventListener('click', async () => {
  await save();
  if (state.saveError || !progress().ready) return;
  state.exportRunning = true; renderSummary();
  try {await api('/api/export', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'}); pollExport();}
  catch (err) {state.exportRunning = false; error(err.message); renderSummary();}
});
async function load() {
  try {
    const data = await api('/api/project');
    state.project = data.project;
    $('source-name').textContent = `${state.project.source.name} · Trinidad`;
    map.setView(state.project.center, 16);
    let draft;
    try {draft = JSON.parse(localStorage.getItem(draftKey()) || 'null');} catch (_) {}
    if (draft && draft.revision === state.project.revision && draft.source.sha256 === state.project.source.sha256) {
      state.project.route = draft.route; state.project.events = draft.events; touch();
    } else if (draft) error('Hay un borrador local de otra revisión. Se abrió la última versión guardada en el equipo.');
    else $('save-state').textContent = 'Guardado en tu equipo';
    renderRoute(); renderMarkers(); renderEvents();
    const first = state.project.events.find(e => e.status === 'pending') || state.project.events[0];
    if (first) {state.selected = first.event_id; renderEvents(); renderEvidence();}
    if (state.project.route.length > 1) $('fit-route').click();
    pollExport();
  } catch (err) {
    $('save-state').textContent = 'Proyecto no disponible'; $('save-state').dataset.state = 'error';
    error(`No se pudo abrir la revisión. ${err.message} Recargá cuando el servidor esté disponible.`);
  }
}
load();
