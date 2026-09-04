/*
  REGISTRO DE DISENOS.
  Para agregar un diseno: copia una entrada, cambia id/nombre/paleta/forma/params.
  Formas disponibles en el motor: 'sphere' 'ripple' 'blob' 'wave' 'dots' 'truchet' 'corners'
  Paletas: libres (4 colores) o las de la app (bQe) - ver bQe en paper-shaders.js.
  El index.html y los ejemplos leen este registro automaticamente.
*/
window.DISENOS = [
  { id: 'aurora', nombre: 'Aurora', paleta: ["#0df2c1", "#0b7cff", "#74efff", "#1a2cff"], forma: 'sphere', speed: 6, softness: .2, intensity: .8, noise: .15 },
  { id: 'ocean',  nombre: 'Ocean',  paleta: ["#b9ecff", "#006494", "#00a6a6", "#072ac8"], forma: 'sphere', speed: 2, softness: .1, intensity: .5, noise: .3 }
];