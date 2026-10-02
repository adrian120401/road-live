# Urban Vision: guía rápida para Windows

## 1. Preparar la laptop una vez

- Instalar **Python 3.11 de 64 bits** y el lanzador `py` del instalador de Python.
- Copiar el proyecto completo a la PC. Crear el entorno allí; no copiar `.venv` de otra máquina.
- Conectar una webcam USB o utilizar la cámara integrada. Una cámara de celular sirve si Windows la reconoce como webcam.
- Activar **Servicios de ubicación** y **Permitir que las aplicaciones accedan a la ubicación** en Configuración > Privacidad y seguridad > Ubicación, según las opciones disponibles en la versión de Windows. El permiso de un sitio en el navegador es independiente.
- En Configuración > Privacidad y seguridad > Cámara, habilitar la cámara y el acceso de aplicaciones de escritorio.
- Mantener la laptop encendida durante el recorrido. La interfaz es una ventana dedicada, sin navegador externo.

Abrir PowerShell en la carpeta del proyecto. Por ejemplo:

```powershell
cd E:\signs\road
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe scripts/download_road_model.py
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

Para NVIDIA compatible con el perfil del proyecto (probado con GTX 1050 Ti), reemplazar
el comando de PyTorch por:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

Comprobar CUDA después de instalar:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
```

La aplicación además prueba un cálculo CUDA real. `--device auto` utiliza CPU si CUDA
no funciona. No requiere instalar Node.js ni el CUDA Toolkit para este perfil.
La instalación inicial y la descarga de modelos requieren internet.

## 2. Ejecutar en vivo

En PowerShell, desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m src.live --device auto
```

Se abre una ventana vertical **9:16**, ajustada al alto de la pantalla. Dejar PowerShell abierto.
La imagen horizontal se recorta en el centro, sin estirarla.

No hace falta activar el entorno ni cambiar la política de ejecución de PowerShell.
El programa invoca el lector nativo de ubicación incluido en el proyecto.
Ejecutar Python de Windows; la ubicación nativa no está disponible con Python de Linux/WSL.

## 3. Hacer el recorrido

1. La cámara se abre automáticamente. **C** abre un selector temporal para elegir otra.
2. **G** elige el modo de ubicación. El modo **Automático** prueba Windows durante 8 s, luego iPhone durante 10 s y, si ambas fuentes fallan, habilita simulación. Las posiciones reales requieren **precisión ≤25 m** y antigüedad **≤5 s**. **Sin GPS** habilita el inicio inmediatamente y usa la ruta simulada de los videos.
3. Pulsar **Iniciar recorrido**, abajo sobre la imagen. Se preparan los modelos y desaparece el botón: quedan únicamente la cámara, detecciones y el HUD original.
4. Al perder ubicación válida aparece **Registro pausado**. La cámara sigue visible; no se agregan ruta, evidencias ni tiempo al MP4. Se reanuda automáticamente al recuperar ubicación.
5. Pulsar **Esc** o **F** para finalizar, incluso durante una pausa. Esperar el guardado del MP4 y las evidencias. El mapa interactivo aparece en **la misma ventana**, con la ruta y los pozos confirmados.
6. **Nuevo recorrido** vuelve a la cámara. **Esc** antes de iniciar cierra la aplicación. Para mover la ventana, arrastrar la parte superior de la imagen.

Cambiar de cámara o fuente de ubicación requiere finalizar el recorrido actual.
La fuente queda fija durante la sesión: perder GPS real pausa el registro; no cambia
a simulación a mitad de un recorrido. El mapa y el JSON rotulan las rutas simuladas.

Para iniciar directamente en modo sin GPS:

```powershell
.\.venv\Scripts\python.exe -m src.live --device auto --no-gps
```

`--location windows` y `--location phone` exigen una posición real de la fuente elegida,
sin fallback simulado. `--location auto` es el comportamiento por defecto.

### Ubicación del iPhone con Camo

La cámara de Camo se elige con **C**, igual que una webcam. El cable transmite la imagen;
no se encontró una API documentada de Camo que exponga la ubicación del iPhone a Python.
La ubicación usa un canal independiente mediante **OwnTracks para iOS**:

1. Instalar [OwnTracks desde la App Store](https://apps.apple.com/us/app/owntracks/id692424691). En iOS, darle ubicación **Siempre** y **Ubicación precisa**, además de acceso a la red local si se solicita.
2. Conectar iPhone y laptop a la misma Wi-Fi. También puede usarse **Compartir Internet del iPhone por USB**, si Windows reconoce el adaptador de red del teléfono. La conexión USB de Camo por sí sola no crea ese canal de red.
3. Abrir la aplicación en Automático o Solo iPhone y pulsar **P** antes de iniciar. Copiar la URL de la interfaz de red compartida; no usar una dirección de VPN u otra red. El receptor escucha en el puerto 8766; `--phone-port 0` elige uno libre.
4. En OwnTracks, elegir conexión **HTTP** e ingresar esa URL. No hace falta un broker MQTT ni un servidor externo. La URL incluye un identificador de emparejamiento y cambia al reiniciar la aplicación.
5. Activar **Move mode**. Ajustar `locatorInterval` a **1 segundo** y `locatorDisplacement` a **1 metro** en los parámetros de OwnTracks. Los valores de fábrica (300 s / 100 m) no sirven para este recorrido.
6. Enviar una ubicación manual para comprobar la conexión. Si Windows solicita acceso de Python a la red, permitir la red privada usada para el teléfono. La URL HTTP se usa en esa red compartida, sin publicar el receptor en internet.
7. Volver a Camo en el iPhone y comprobar que OwnTracks siga enviando posiciones en segundo plano. El modo de ubicación de la laptop debe indicar **iPhone**. El sistema comprueba precisión y tiempo originales de cada mensaje; retrasos de iOS, señal insuficiente o falta de conexión producen pausa en un recorrido real.

Si OwnTracks no se detecta, comprobar la configuración exportada:
`mode: 3` es HTTP, `monitoring: 2` es Move, `locatorInterval: 1` y
`locatorDisplacement: 1`. `monitoring: 1` es cambios significativos y no sirve para
posiciones cada pocos segundos. No confundir la configuración (`_type: configuration`)
o los waypoints con un mensaje de ubicación (`_type: location`, con `lat`, `lon`, `acc`, `tst`).

En la versión actualizada, abrir la URL completa que muestra **P** en Safari del iPhone
devuelve un diagnóstico sin coordenadas. `online: true` confirma conexión al receptor;
`messages_received: 0` indica que todavía no llegaron mensajes de OwnTracks.
`last_error` muestra mensajes rechazados y `location` indica precisión y antigüedad.
Si Safari no conecta, revisar la Wi-Fi compartida, acceso a red local en iOS y firewall
de Python en Windows. Si responde 403, la URL/token no corresponde a la ejecución actual.
Si el recorrido ya comenzó en simulación, finalizarlo y crear otro para usar el iPhone:
la fuente queda fija dentro de cada recorrido. **Solo iPhone**, seleccionado con **G**,
facilita la prueba porque no cambia a simulación mientras se configura el teléfono.

La recepción y el protocolo se verificaron con mensajes de prueba; la combinación
OwnTracks en segundo plano + Camo debe comprobarse en el iPhone concreto. No se garantiza
que iOS mantenga lecturas cada segundo. Si no cumple la precisión/frecuencia, usar Sin GPS.
La ruta simulada se ajusta al tiempo registrado al finalizar y no representa posiciones reales.

Referencias: [Camo por USB](https://camo.com/support/camo/camo-getting-started),
[OwnTracks HTTP](https://owntracks.org/booklet/tech/http/) y
[Move mode en iOS](https://owntracks.org/booklet/features/location/).

## 4. Resultados y ubicación

Las sesiones tienen carpetas independientes para conservar recorridos anteriores:

```text
outputs/live/<fecha_id>/
  recorrido.mp4
  map.html
  recorrido.json
  recorrido_events.json
  recorrido_events/pothole_001.jpg
```

El MP4 incluye la cámara, detecciones y el HUD, en **1080×1920 a 30 FPS**, codec
`mp4v`, sin audio. Los controles y el mapa no se graban. Cuando la inferencia es más
lenta que 30 FPS, se repite la última imagen para mantener la duración real; eso no
crea detecciones nuevas. La grabación comienza con la primera imagen analizada.
Las pausas se excluyen del clip. Cada evidencia incluye `video_timestamp` (segundos)
y `video_frame` (índice desde cero) para ubicarla en el MP4.

Los segmentos separados por pausas se dibujan sin conectar los huecos; la distancia
estimada solo suma los tramos registrados. Una sesión sin imágenes analizadas conserva
el mapa/resumen vacío y no genera MP4.

La ubicación se consulta directamente al servicio de Windows, no al navegador.
**Tener una laptop no implica tener GPS satelital.** Windows puede utilizar GPS, Wi-Fi
o IP, y la posición puede ser demasiado imprecisa para iniciar. Si Windows devuelve
±141 m, por ejemplo, Solo Windows permanecerá esperando; Automático probará iPhone y luego simulación.
Para mayor precisión se necesita un proveedor de ubicación adecuado, como GNSS reconocido
por Windows. Un receptor USB que solo entrega datos por puerto serie no se integra automáticamente.

### Diagnosticar la ubicación

Ejecutar en la laptop donde ocurre el problema; puede hacerse con el navegador cerrado:

```powershell
.\.venv\Scripts\python.exe scripts/check_location.py --seconds 15
```

El diagnóstico utiliza exactamente el lector nativo de la aplicación
(`System.Device.Location.GeoCoordinateWatcher`, precisión solicitada `High`).
No imprime coordenadas. Informa:

- `permission`: `Granted` confirma permiso para el lector nativo; `Denied` requiere revisar los permisos de Windows.
- `accuracy_m`: incertidumbre horizontal en metros. ±100 m no significa que se haya denegado permiso.
- `age_seconds`: tiempo desde la posición original. Leer repetidamente una posición almacenada no la vuelve reciente.
- `valid`: requiere precisión ≤25 m y antigüedad ≤5 s, incluso con permiso concedido.
- `position_source: Unknown`: esta API no expone qué proveedor físico entregó la posición. No se etiqueta una estimación como GPS.

Si aparece `Granted` con margen alto, comprobar en la documentación del modelo si
la laptop incluye receptor GPS/GNSS. Si lo incluye, revisar su controlador y su
integración con el servicio de ubicación de Windows; probar donde tenga recepción
satelital. Un permiso de navegador no habilita ni agrega un receptor físico.
Para analizar el problema en otra laptop, conservar la salida del diagnóstico y
anotar el modelo de equipo y la versión de Windows.

Referencias: [servicio de ubicación y privacidad de Windows](https://support.microsoft.com/en-us/windows/privacy/windows-location-service-and-privacy)
y [selección de proveedores de GeoCoordinateWatcher](https://learn.microsoft.com/en-us/dotnet/api/system.device.location.geocoordinatewatcher?view=netframework-4.8.1).

Los marcadores indican la posición aproximada del vehículo al capturar la evidencia,
con el margen de error informado por Windows; no la posición física exacta del pozo.
La distancia y los conteos son estimados. El fondo OpenStreetMap requiere internet;
la ruta y las evidencias siguen disponibles sin el fondo de calles.

Desconectar la cámara, cerrar la ventana con **Alt+F4** o pulsar **Ctrl+C** guarda un
recorrido parcial. No cerrar la terminal a la fuerza durante el guardado.
Si falla una exportación, la ventana informa el error y presenta el mapa si se pudo generar.

El encuadre de la cámara importa: los defaults de ROI/calzada pertenecen a los videos
revisados del proyecto. Para ajustar la zona de búsqueda:

```powershell
.\.venv\Scripts\python.exe -m src.live --road-roi 0.05 0.45 0.95 0.70
```

Los cuatro valores son izquierda, arriba, derecha y abajo normalizados entre 0 y 1.
`--no-road-area` desactiva el filtro de veredas cuando la geometría preconfigurada
no corresponde a la nueva cámara; puede aumentar falsos positivos. El rendimiento real
depende de la CPU/GPU, la iluminación y el encuadre; la vista procesa imágenes recientes
sin acumular una cola de video.

## 5. Probar con un video grabado

```powershell
.\.venv\Scripts\python.exe src/main.py --input video2.MOV --road-damage --show --road-roi 0.05 0.48 0.95 0.70 --output outputs/prueba.mp4
.\.venv\Scripts\python.exe scripts/open_map.py --map outputs/prueba_map.html
```

Este modo obtiene ubicación de los metadatos del archivo o usa la ruta simulada indicada
en el mapa. No consulta la ubicación actual de la laptop para geolocalizar un video pasado.

Para ejecutar las verificaciones automatizadas:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
