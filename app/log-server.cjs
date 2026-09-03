const http = require('http');
const fs = require('fs');
const path = require('path');
const logFile = path.join(__dirname, 'browser-errors.log');
try { fs.writeFileSync(logFile, `# browser-errors.log iniciado ${new Date().toISOString()}\n`, 'utf-8'); console.log(`[log-server] ${logFile}`);} catch(e){ console.error(e); }
const server = http.createServer((req, res) => {
  const url = req.url || '';
  if (url.startsWith('/__browser-log')) {
    if (req.method === 'POST' && url.startsWith('/__browser-log-clear')) {
      try { fs.writeFileSync(logFile, `# cleared ${new Date().toISOString()}\n`, 'utf-8'); } catch {}
      res.writeHead(200, {'Content-Type':'text/plain'}); res.end('cleared'); return;
    }
    if (req.method === 'POST') {
      let body=''; req.on('data', c=> body+=c); req.on('end', ()=>{
        try { const data=JSON.parse(body); const line=data.entry||JSON.stringify(data); fs.appendFileSync(logFile, line+'\n','utf-8'); console.log('[BROWSER-LOG]', line);} catch { fs.appendFileSync(logFile, `[parse-error] ${body}\n`,'utf-8'); }
        res.writeHead(200, {'Content-Type':'text/plain'}); res.end('ok');
      }); return;
    }
    if (req.method === 'GET') {
      try { const content=fs.readFileSync(logFile,'utf-8'); res.writeHead(200, {'Content-Type':'text/plain; charset=utf-8'}); res.end(content);} catch { res.writeHead(200, {'Content-Type':'text/plain'}); res.end('# no log yet\n'); }
      return;
    }
  }
  res.writeHead(404); res.end('not found');
});
server.listen(3001, '127.0.0.1', ()=> console.log('[log-server] listening http://127.0.0.1:3001'));
