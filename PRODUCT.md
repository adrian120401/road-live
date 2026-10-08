# Urban Vision

Herramienta local de análisis urbano para recorridos grabados desde un vehículo.
El usuario revisa evidencias de pozos y reconstruye el recorrido en Trinidad, Uruguay,
para producir un video con HUD y un mapa de los daños revisados.

## Plataforma y stack

Python en Windows, YOLO local, OpenCV, FFmpeg. El editor funciona en el navegador
mediante HTML, CSS, JavaScript y Leaflet, servido exclusivamente por loopback.
Se extiende el visor y la identidad visual existentes; no hay cuentas ni base de datos.

## Flujo confirmado

Priorizar análisis offline y calidad de imagen. Dibujar inicio, esquinas, giros y fin;
revisar las fotos, confirmar y ubicar detecciones o descartarlas. No agregar pozos
manualmente, unir duplicados ni sincronizar tiempos de los giros en esta versión.
Guardar la revisión separada de las detecciones originales y reabrirla.
Exportar 2160×3840 a 60 FPS, sin audio, con ocho segundos de mapa al cierre.
Conservar autos, personas, motos y las demás capas activas; excluir camión y bicicleta.

## Restricciones

No presentar coordenadas manuales como GPS ni la ruta de respaldo como recorrido real.
No inventar las calles tomadas ni las posiciones de los pozos. El usuario las aporta
mediante el editor. La exportación final requiere revisión completa y ruta dibujada.
El mapa de calles requiere internet; los resultados y la edición se guardan localmente.
