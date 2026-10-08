# Editor de recorrido de Urban Vision

Extensión del visor existente, destinada a operar: dibujar calles y ubicar detecciones.
Conservar la identidad de Urban Vision, el fondo oscuro y la tipografía Arial existente.

## Sistema visual

- Fondo `#171e22`, panel `#20292d`, controles `#29353a`, texto `#eaf1ed`.
- Texto secundario `#b5c4c8`, límites `#415158`, ruta y acción principal `#8be6ce`.
- Pozos `#ecc17b`, errores `#ffc0b5`. Radios de controles de 7px; foco visible de 2px.
- Encabezado de 82px; mapa flexible y panel de revisión de 390px. Por debajo de 720px,
  el mapa precede a la revisión en una columna sin desplazamiento horizontal.
- La foto contextual del pozo precede a la foto completa. Los estados de revisión
  usan texto además de color. Contadores, coordenadas y tiempos usan números tabulares.

## Interacciones

Dos modos explícitos: dibujar recorrido y ubicar pozos. Ningún clic de revisión agrega
puntos de ruta. Guardado automático con indicador persistente y borrador local ante
errores de conexión. La exportación requiere revisión guardada y completa.
Los movimientos responden a la acción del usuario; respetar movimiento reducido.
