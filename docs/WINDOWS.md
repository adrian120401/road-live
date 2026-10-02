# Urban Vision: guía rápida para Windows

## 1. Preparar la laptop una vez

- Instalar **Python 3.11 de 64 bits** y el lanzador `py` del instalador de Python.
- Copiar el proyecto completo a la PC. Crear el entorno allí; no copiar `.venv` de otra máquina.
- Conectar una webcam USB o utilizar la cámara integrada. Una cámara de celular sirve si Windows la reconoce como webcam.
- Activar **Ubicación**, **Cámara** y el acceso de aplicaciones de escritorio en Configuración > Privacidad y seguridad. Los nombres pueden variar según la versión de Windows.
- Usar Edge o Chrome para abrir la interfaz local. Mantener la laptop encendida y con la página visible durante el recorrido.

Abrir PowerShell en la carpeta del proyecto. Por ejemplo:

```powershell
cd E:\signs\road
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
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

El navegador abre `http://127.0.0.1:8765/`. Dejar PowerShell abierto. Si ese puerto está ocupado:

```powershell
.\.venv\Scripts\python.exe -m src.live --device auto --port 0
```

No hace falta activar el entorno ni cambiar la política de ejecución de PowerShell.
El programa invoca el lector nativo de ubicación incluido en el proyecto.
Ejecutar Python de Windows; la ubicación nativa no está disponible con Python de Linux/WSL.

## 3. Hacer el recorrido

1. Pulsar **Habilitar cámara** y aceptar el permiso del navegador. Elegir la webcam en el selector.
2. El botón **Cámara**, o la tecla **C** fuera de un campo de formulario, abre/cierra el selector.
3. Esperar una ubicación de Windows con **precisión de 25 m o mejor**, de hasta **5 segundos** de antigüedad. No hay simulación automática.
4. Pulsar **Iniciar recorrido**. La primera imagen prepara los modelos; luego aparecen las detecciones, el tiempo, los FPS de procesamiento y el contador estimado de pozos.
5. Si la ubicación deja de cumplir los requisitos, aparece **Registro pausado**: la cámara sigue visible, pero no se registran ruta, detecciones ni nuevas evidencias. Se reanuda automáticamente al recuperar una posición válida.
6. Pulsar **Finalizar recorrido**, también disponible durante una pausa. Esperar **Guardando recorrido…**. Se abre automáticamente el mapa con la ruta registrada, inicio/fin y pozos confirmados con fotos y confianza.
7. **Nuevo recorrido** vuelve a la preparación para elegir cámara e iniciar otra sesión.

Cambiar de cámara requiere finalizar el recorrido actual.

## 4. Resultados y ubicación

Las sesiones tienen carpetas independientes para conservar recorridos anteriores:

```text
outputs/live/<fecha_id>/
  map.html
  recorrido.json
  recorrido_events.json
  recorrido_events/pothole_001.jpg
```

El modo en vivo guarda datos, mapa y fotos; no graba MP4. El modo de archivos conserva
su exportación de video. Los tramos separados por pausas se dibujan sin conectar los huecos,
y la distancia estimada solo suma los segmentos registrados.

La ubicación se consulta directamente al servicio de Windows, no al navegador.
**Tener una laptop no implica tener GPS satelital.** Windows puede utilizar GPS, Wi-Fi
o IP, y la posición puede ser demasiado imprecisa para iniciar. Si Windows devuelve
±141 m, por ejemplo, el programa permanecerá esperando: requiere ±25 m o mejor.
Para mayor precisión se necesita un proveedor de ubicación adecuado, como GNSS reconocido
por Windows. Un receptor USB que solo entrega datos por puerto serie no se integra automáticamente.

Los marcadores indican la posición aproximada del vehículo al capturar la evidencia,
con el margen de error informado por Windows; no la posición física exacta del pozo.
La distancia y los conteos son estimados. El fondo OpenStreetMap requiere internet;
la ruta y las evidencias siguen disponibles sin el fondo de calles.

Cerrar la cámara/página, perder conexión con la página durante más de 10 s o presionar
Ctrl+C guarda un recorrido parcial. No cerrar la terminal a la fuerza durante el guardado.
Si hay un error de exportación, la página informa qué falló y muestra el mapa si se pudo generar.

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
