import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR se desactiva con DISABLE_HMR=true. No modificar: evita parpadeos durante ediciones.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Sin file watching cuando DISABLE_HMR=true para ahorrar CPU.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
      proxy: {
        '/__browser-log': 'http://127.0.0.1:3001',
        '/__browser-log-clear': 'http://127.0.0.1:3001',
        '/api/prediccion-frame': {
          target: 'http://127.0.0.1:8765',
          changeOrigin: true,
        },
        '/api/prediccion': {
          target: 'http://127.0.0.1:8765',
          changeOrigin: true,
        },
      },
    },
  };
});
