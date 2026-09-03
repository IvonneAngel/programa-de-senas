import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// ponytail: cerebro siempre prendido — captura todo lo que dice el navegador y lo guarda
// No se entrega nada si hay errores; se lee, se guarda y se revisa.
const BROWSER_LOG_KEY = '__browser_errors__';
function guardarError(tipo: string, mensaje: string, extra?: string) {
  const entry = `[${new Date().toISOString()}] [${tipo}] ${mensaje}${extra ? ' | ' + extra : ''}`;
  try {
    const prev = JSON.parse(sessionStorage.getItem(BROWSER_LOG_KEY) || '[]');
    prev.push(entry);
    sessionStorage.setItem(BROWSER_LOG_KEY, JSON.stringify(prev.slice(-50)));
    // eslint-disable-next-line no-console
    console.log('[BROWSER-LOG]', entry);
  } catch {}
  // enviar al servidor vite para que quede en archivo físico
  try {
    fetch('/__browser-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tipo, mensaje, extra, entry, url: location.href }),
    }).catch(() => {});
  } catch {}
  // exponer global para inspección manual desde Zen console
  (window as unknown as Record<string, unknown>).__getBrowserErrors = () =>
    JSON.parse(sessionStorage.getItem(BROWSER_LOG_KEY) || '[]');
  (window as unknown as Record<string, unknown>).__clearBrowserErrors = () => {
    sessionStorage.removeItem(BROWSER_LOG_KEY);
    fetch('/__browser-log-clear', { method: 'POST' }).catch(() => {});
  };
}

window.addEventListener('error', (e) => {
  guardarError('error', e.message, `${e.filename}:${e.lineno}:${e.colno} | ${e.error?.stack || ''}`);
});
window.addEventListener('unhandledrejection', (e: PromiseRejectionEvent) => {
  const msg = e.reason?.message || String(e.reason);
  const stack = e.reason?.stack || '';
  guardarError('unhandledrejection', msg, stack);
});
const _origError = console.error;
console.error = (...args: unknown[]) => {
  _origError(...args);
  try {
    const msg = args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
    guardarError('console.error', msg);
  } catch {}
};
const _origWarn = console.warn;
console.warn = (...args: unknown[]) => {
  _origWarn(...args);
  try {
    const msg = args.map((a) => (typeof a === 'string' ? a : String(a))).join(' ');
    if (msg.includes('Failed') || msg.includes('Error') || msg.includes('404') || msg.includes('WebGL')) {
      guardarError('console.warn', msg);
    }
  } catch {}
};

// log de arranque para confirmar logger activo
guardarError('info', 'browser logger activo', navigator.userAgent);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
