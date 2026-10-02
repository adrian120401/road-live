'use strict';
const el = id => document.getElementById(id);
const video = el('camera-video'), processed = el('processed-image');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
let stream = null, session = null, gps = null, polling = false, starting = false;
let capturing = false, stopping = false, imageURL = null, resultsShown = false;
let switching = false;
let dismissedSessionId = null;
const active = () => session && !['finished', 'error'].includes(session.state);
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
function error(message) { el('error').textContent = message; el('error').hidden = !message; }
async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Error de conexión (${response.status}).`;
    try { message = (await response.json()).error || message; } catch (_) {}
    throw new Error(message);
  }
  return response;
}
function cameraPanel(show) {
  if (active() || starting || resultsShown) return;
  el('camera-panel').hidden = !show;
  el('camera-toggle').setAttribute('aria-expanded', String(show));
  if (show && !el('camera-select').disabled) el('camera-select').focus();
}
el('camera-toggle').addEventListener('click', () => cameraPanel(el('camera-panel').hidden));
document.addEventListener('keydown', event => {
  if (event.key.toLowerCase() === 'c' && !event.ctrlKey && !event.metaKey && !event.altKey &&
      !['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName) && !event.target.isContentEditable) {
    event.preventDefault(); cameraPanel(el('camera-panel').hidden);
  }
});
async function listCameras() {
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === 'videoinput');
  const current = stream?.getVideoTracks()[0]?.getSettings().deviceId;
  el('camera-select').replaceChildren(...cameras.map((device, i) => {
    const option = new Option(device.label || `Cámara ${i + 1}`, device.deviceId);
    option.selected = device.deviceId === current;
    return option;
  }));
  el('camera-select').disabled = !cameras.length || active();
}
async function connectCamera(deviceId) {
  if (active() || starting || switching) return;
  switching = true; el('camera-enable').disabled = true;
  try {
    error('');
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Abrí la página local en Edge o Chrome para acceder a la cámara.');
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {
      ...(deviceId ? {deviceId: {exact: deviceId}} : {}),
      width: {ideal: 1280}, height: {ideal: 720}, frameRate: {ideal: 30}
    }});
    video.srcObject = stream;
    await video.play();
    if (!video.videoWidth) await new Promise(resolve => video.addEventListener('loadedmetadata', resolve, {once: true}));
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      if (active()) finishTrip(true);
      error('La cámara se desconectó. Reconectala para comenzar otro recorrido.');
      stream = null; render();
    });
    el('placeholder').hidden = true; video.hidden = false; processed.hidden = true;
    el('camera-label').textContent = stream.getVideoTracks()[0].label || 'CÁMARA CONECTADA';
    el('camera-message').textContent = 'Lista para registrar. Usá C para abrir o cerrar este selector.';
    el('camera-enable').textContent = 'Actualizar cámaras';
    await listCameras();
  } catch (exc) {
    stream = null; video.hidden = true; el('placeholder').hidden = false;
    const messages = {NotAllowedError: 'Permiso de cámara denegado. Habilitalo en el navegador y en Windows.',
      NotFoundError: 'No hay una webcam disponible. Conectá una cámara y volvé a intentar.',
      NotReadableError: 'No se pudo abrir la cámara. Comprobá que otra aplicación no la esté usando.'};
    error(messages[exc.name] || exc.message);
  } finally {
    switching = false; render();
  }
}
el('camera-enable').addEventListener('click', () => stream ? listCameras().catch(exc => error(exc.message)) : connectCamera());
el('camera-select').addEventListener('change', () => connectCamera(el('camera-select').value));
navigator.mediaDevices?.addEventListener('devicechange', () => { if (!active() && stream) listCameras().catch(exc => error(exc.message)); });
function render() {
  if (gps) {
    el('gps-dot').classList.toggle('valid', gps.valid);
    el('gps-reason').textContent = gps.valid ? `Posición válida · actualizada hace ${gps.age_seconds.toFixed(1)} s` : gps.reason;
    el('gps-accuracy').textContent = gps.accuracy_m === null ? '—' : `±${Math.round(gps.accuracy_m)} m`;
  }
  const busy = active() || starting;
  el('camera-toggle').disabled = busy || resultsShown;
  el('camera-toggle').hidden = resultsShown;
  el('camera-enable').disabled = busy || switching;
  el('camera-select').disabled = busy || switching || !stream;
  el('start').hidden = !!busy;
  el('start').disabled = !stream || !gps?.valid || switching || starting;
  el('finish').hidden = !busy;
  el('finish').disabled = stopping || session?.state === 'finishing' || starting;
  el('finish').firstChild.textContent = stopping || session?.state === 'finishing' ? 'Guardando recorrido… ' : 'Finalizar recorrido ';
  const titles = {preparing: 'Preparando modelos', running: 'Recorrido en curso', paused: 'Registro pausado',
    finishing: 'Guardando recorrido', finished: 'Recorrido finalizado', error: 'Recorrido interrumpido'};
  el('trip-state').textContent = titles[session?.state] || 'Preparar recorrido';
  el('trip-reason').textContent = session?.reason || 'Habilitá una cámara y esperá una ubicación válida.';
  el('screen-badge').hidden = session?.state !== 'paused';
  if (session?.state === 'paused') { video.hidden = !stream; processed.hidden = true; }
  const seconds = Math.floor(session?.elapsed_seconds || 0);
  el('duration').textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  el('potholes').textContent = session?.potholes || 0;
  el('fps').replaceChildren(document.createTextNode(`${session?.processing_fps || '—'} `), Object.assign(document.createElement('small'), {textContent: 'FPS'}));
  if (session && ['finished', 'error'].includes(session.state)) showResult();
}
function showResult() {
  stopping = false;
  const messages = [session.error, ...(session.export_errors || [])].filter(Boolean);
  if (messages.length) error(messages.join(' · '));
  if (resultsShown) return;
  resultsShown = true;
  el('camera-toggle').hidden = true;
  el('page-title').textContent = 'El recorrido, en el mapa.';
  el('page-intro').textContent = 'Ruta registrada y evidencias de cada hallazgo.';
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
  el('capture-layout').hidden = true; el('camera-panel').hidden = true; el('results').hidden = false;
  el('result-title').textContent = session.reason === 'Recorrido finalizado' ? 'Recorrido registrado.' : 'Recorrido parcial guardado.';
  el('result-summary').textContent = `${el('duration').textContent} de recorrido · ${session.potholes} pozos estimados · Ubicación de Windows`;
  el('download-summary').href = session.summary_url || '#';
  el('download-summary').hidden = !session.summary_url;
  if (session.map_url) { el('map').src = session.map_url; el('map').hidden = false; }
  else { el('map').hidden = true; error(messages.join(' · ') || 'No se pudo generar el mapa. Los datos guardados están en la carpeta del recorrido.'); }
  el('new-trip').focus();
}
async function poll() {
  if (polling) return;
  polling = true;
  try {
    const requestedId = session?.id;
    const data = await (await request(`/api/state${requestedId ? `?session=${requestedId}` : ''}`)).json();
    gps = data.location;
    if (requestedId !== session?.id) return; // Ignore a poll from a previous start/reset.
    if (session && data.session?.id !== session.id && active()) throw new Error('El servidor se reinició. Revisá el recorrido guardado antes de comenzar otro.');
    if (data.session && data.session.id !== dismissedSessionId && (!session || data.session.id === session.id)) session = data.session;
    render();
  } catch (exc) { error(`No se pudo consultar el programa: ${exc.message}`); }
  finally { polling = false; }
}
async function captureLoop() {
  if (capturing) return;
  capturing = true;
  const tripId = session.id;
  try {
    while (active() && session.id === tripId && !stopping && stream) {
      const began = performance.now();
      const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
      canvas.width = Math.max(2, Math.round(video.videoWidth * scale / 2) * 2);
      canvas.height = Math.max(2, Math.round(video.videoHeight * scale / 2) * 2);
      const captureTime = Date.now();
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .85));
      if (!blob) throw new Error('No se pudo capturar la imagen de la cámara.');
      const response = await request(`/api/${tripId}/frames`, {method: 'POST',
        headers: {'Content-Type': 'image/jpeg', 'X-Capture-Time': String(captureTime)}, body: blob});
      if (response.status === 200 && !stopping && active() && session.id === tripId && session.state !== 'paused') {
        const nextURL = URL.createObjectURL(await response.blob());
        const previousURL = imageURL;
        processed.onload = () => { if (previousURL) URL.revokeObjectURL(previousURL); };
        processed.src = nextURL; imageURL = nextURL;
        processed.hidden = false; video.hidden = true;
        el('view-label').textContent = 'DETECCIONES EN VIVO';
      }
      await wait(Math.max(0, 100 - (performance.now() - began)));
    }
  } catch (exc) {
    if (!stopping && session?.id === tripId) { error(exc.message); if (active()) await finishTrip(true); }
  } finally {
    capturing = false;
    if (active() && stream && !stopping && session.id !== tripId) captureLoop();
  }
}
el('start').addEventListener('click', async () => {
  if (!stream || !gps?.valid || active() || starting) return;
  starting = true; error(''); render();
  try {
    session = await (await request('/api/start', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({camera_name: stream.getVideoTracks()[0].label})})).json();
    el('camera-panel').hidden = true;
    el('camera-toggle').setAttribute('aria-expanded', 'false');
    stopping = false; render(); captureLoop();
  } catch (exc) { error(exc.message); }
  finally { starting = false; render(); }
});
async function finishTrip(disconnected = false) {
  if (!active() || stopping) return;
  stopping = true; render();
  try {
    session = await (await request(`/api/${session.id}/${disconnected ? 'disconnect' : 'finish'}`, {method: 'POST'})).json();
    await poll();
  } catch (exc) { stopping = false; error(`No se pudo finalizar: ${exc.message}`); render(); }
}
el('finish').addEventListener('click', () => finishTrip());
el('new-trip').addEventListener('click', () => {
  dismissedSessionId = session?.id || null;
  session = null; resultsShown = false; stopping = false;
  el('map').removeAttribute('src'); el('results').hidden = true; el('capture-layout').hidden = false;
  el('camera-panel').hidden = false; el('camera-toggle').setAttribute('aria-expanded', 'true');
  el('camera-enable').textContent = 'Habilitar cámara';
  el('page-title').textContent = 'El recorrido empieza acá.';
  el('page-intro').textContent = 'Elegí tu cámara. Registrá la calle. Revisá cada hallazgo en el mapa.';
  video.hidden = true; processed.hidden = true; el('placeholder').hidden = false;
  el('view-label').textContent = 'VISTA DE CÁMARA'; el('camera-label').textContent = 'SIN CONECTAR';
  if (imageURL) { URL.revokeObjectURL(imageURL); imageURL = null; }
  error(''); render(); el('camera-enable').focus();
});
window.addEventListener('pagehide', () => {
  if (active() && !stopping) navigator.sendBeacon(`/api/${session.id}/disconnect`, '');
});
setInterval(poll, 500);
poll();
