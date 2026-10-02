// Optional end-to-end map audit using a local Chrome DevTools session.
const fs = require('node:fs');
const target = process.argv[2] || '/outputs/map.html';
const port = process.argv[3] || '9339';
const prefix = process.argv[4] || 'output/playwright/v31-map';

(async () => {
  const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const page = pages.find(p => p.url.includes(target));
  if (!page) throw Error(`Map tab missing: ${target}`);
  const socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise(resolve => socket.addEventListener('open',resolve,{once:true}));
  let id = 0;
  const pending = new Map(), requests = new Map(), headers = new Map(), network = [], errors = [];
  socket.addEventListener('message',message => {
    const data = JSON.parse(message.data);
    if (data.id) {
      const job = pending.get(data.id); pending.delete(data.id);
      data.error ? job.reject(data.error) : job.resolve(data.result);
    } else if (data.method === 'Network.requestWillBeSent') requests.set(data.params.requestId,data.params.request.url);
    else if (data.method === 'Network.requestWillBeSentExtraInfo') headers.set(data.params.requestId,data.params.headers);
    else if (data.method === 'Network.responseReceived' && data.params.response.url.startsWith('https://tile.openstreetmap.org/'))
      network.push({id:data.params.requestId,status:data.params.response.status,cached:data.params.response.fromDiskCache});
    else if (data.method === 'Runtime.exceptionThrown') errors.push(data.params.exceptionDetails.text);
  });
  function send(method,params={}) {
    return new Promise((resolve,reject) => {const key=++id;pending.set(key,{resolve,reject});socket.send(JSON.stringify({id:key,method,params}));});
  }
  async function evaluate(expression) {
    const result = await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  async function screenshot(suffix) {
    const shot = await send('Page.captureScreenshot',{format:'png'});
    fs.writeFileSync(prefix+suffix+'.png',Buffer.from(shot.data,'base64'));
  }
  await send('Network.enable'); await send('Runtime.enable');
  await send('Emulation.clearDeviceMetricsOverride'); await send('Page.reload');
  await new Promise(resolve=>setTimeout(resolve,3500));
  const state = JSON.parse(await evaluate(`JSON.stringify({url:location.href,events:data.events.length,buttons:document.querySelectorAll('.event').length,summary:document.querySelector('.summary').innerText,tiles:[...document.querySelectorAll('.leaflet-tile')].map(i=>({loaded:i.complete&&i.naturalWidth>0,policy:i.referrerPolicy})),warningVisible:getComputedStyle(document.getElementById('tile-warning')).display!=='none'})`));
  await screenshot('');
  await evaluate("document.querySelector('.event').click()");
  await new Promise(resolve=>setTimeout(resolve,700));
  const popup = JSON.parse(await evaluate(`JSON.stringify({text:document.querySelector('.leaflet-popup-content').innerText,imageLoaded:document.querySelector('.evidence').naturalWidth>0,link:document.querySelector('.photo-link').href,rect:document.querySelector('.leaflet-popup').getBoundingClientRect().toJSON(),panel:document.querySelector('.summary').getBoundingClientRect().toJSON()})`));
  await screenshot('-popup');
  const photos = await evaluate(`Promise.all(data.events.map(async e=>{const wrapper=document.createElement('div');wrapper.innerHTML=e.popup;const img=wrapper.querySelector('img');const link=wrapper.querySelector('.photo-link');const response=await fetch(link.getAttribute('href'));return {id:e.id,embedded:img?.src.startsWith('data:image/jpeg;base64,'),originalStatus:response.status}}))`);
  await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate('map.invalidateSize();map.closePopup();document.querySelector(".event").click()');
  await new Promise(resolve=>setTimeout(resolve,700));
  const mobile = JSON.parse(await evaluate(`JSON.stringify({popup:document.querySelector('.leaflet-popup').getBoundingClientRect().toJSON(),panel:document.querySelector('.summary').getBoundingClientRect().toJSON(),imageLoaded:document.querySelector('.evidence').naturalWidth>0})`));
  await screenshot('-mobile');
  const report = {state,popup,photos,mobile,errors,tiles:network.map(n=>({url:requests.get(n.id),status:n.status,cached:n.cached,referer:headers.get(n.id)?.Referer}))};
  fs.writeFileSync(prefix+'-verification.json',JSON.stringify(report,null,2));
  if (errors.length || photos.some(p=>!p.embedded || p.originalStatus!==200) || !popup.imageLoaded || !mobile.imageLoaded)
    throw Error('Map/photo verification failed');
  console.log(JSON.stringify({events:state.events,photoLinks:photos.length,
    tilesLoaded:state.tiles.every(t=>t.loaded),tileStatuses:[...new Set(report.tiles.map(t=>t.status))],
    desktopImage:popup.imageLoaded,mobileImage:mobile.imageLoaded,errors},null,2));
  socket.close();
})().catch(error=>{console.error(error);process.exit(1)});
