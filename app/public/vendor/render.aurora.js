/* Render compartido: usa el motor de paper-shaders.js + el registro disenos.js. */
function bolitaWrap(d, px) {
  return a.jsx('div', { style: { width: px, height: px, borderRadius: '50%', overflow: 'hidden', flex: 'none' }, children:
    a.jsx($R, { speed: d.speed, softness: d.softness, intensity: d.intensity, noise: d.noise, shape: d.forma, colors: d.paleta, colorBack: '#ffffff00' })
  });
}
function renderGaleria(root) {
  const items = window.DISENOS.map(d => a.jsxs('div', { className: 'card', key: d.id, children: [
    bolitaWrap(d, 40),
    a.jsxs('div', { className: 'meta', children: [
      a.jsx('div', { className: 'nombre', children: d.nombre }),
      a.jsx('div', { className: 'forma', children: 'forma: ' + d.forma + ' | speed ' + d.speed + ' | softness ' + d.softness + ' | intensity ' + d.intensity + ' | noise ' + d.noise })
    ]})
  ]}));
  ReactDOM.createRoot(root).render(a.jsx('div', { className: 'gallery' }, ...items));
  fixShaderSize();
}
function renderEspecimen(root, id) {
  const d = window.DISENOS.find(x => x.id === id);
  if (!d) { throw new Error('diseno no encontrado en DISENOS: ' + id); }
  ReactDOM.createRoot(root).render(a.jsxs('div', { className: 'spec', children: [
    bolitaWrap(d, 128),
    a.jsxs('div', { className: 'ficha', children: [
      a.jsx('h1', { children: d.nombre }),
      a.jsxs('p', { className: 'param', children: ['forma: ' + d.forma + '  |  speed ' + d.speed + '  |  softness ' + d.softness + '  |  intensity ' + d.intensity + '  |  noise ' + d.noise] }),
      a.jsxs('div', { className: 'paleta', children: d.paleta.map(c => a.jsx('span', { className: 'swatch', style: { background: c }, key: c })) })
    ]})
  ]}));
  fixShaderSize();
}
function fixShaderSize() {
  document.querySelectorAll('#root [data-paper-shader]').forEach(el => {
    const m = el.paperShaderMount;
    if (!m) return;
    // si el ResizeObserver aun no notifico (pestana oculta), alimentar los mismos
    // numeros que daria el RO: tamano del elemento x devicePixelRatio
    if (!m.devicePixelsSupported) {
      const r = el.getBoundingClientRect();
      if (r.width > 0) {
        const dpr = Math.max(1, window.devicePixelRatio);
        m.devicePixelsSupported = true;
        m.parentWidth = r.width;
        m.parentHeight = r.height;
        m.parentDevicePixelWidth = r.width * dpr;
        m.parentDevicePixelHeight = r.height * dpr;
      }
    }
    m.handleResize();
  });
}
document.addEventListener('visibilitychange', fixShaderSize);
setTimeout(fixShaderSize, 100);
setInterval(fixShaderSize, 1000);