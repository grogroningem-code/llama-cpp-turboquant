#!/usr/bin/env python3
"""Option-1 server A/B leg: launch one server binary, replay fixed prompts, log pp/tg/acceptance.
Usage: bench-server-ab.py <server-exe> <model> <port> <sampling-json> <prompts-json> <out-json> [-- <extra server flags...>]
Prompts json: list of {id, text}. Each prompt: 1 warm-up (discarded) per leg + measured run.
"""
import json, subprocess, sys, time, urllib.request

srv_exe, model, port, samp_f, prom_f, out_f = sys.argv[1:7]
extra = sys.argv[8:] if len(sys.argv) > 8 and sys.argv[7] == '--' else sys.argv[7:]
samp = json.load(open(samp_f))
prompts = json.load(open(prom_f))
base = f'http://localhost:{port}'

def req(path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.urlopen(urllib.request.Request(base + path, data=data,
        headers={'Content-Type': 'application/json'}), timeout=timeout)
    return json.loads(r.read())

def gpu_idle():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu',
            '--format=csv,noheader'], capture_output=True, text=True, timeout=30).stdout
        u, t = o.strip().split(',')
        return float(u.split()[0]), float(t.split()[0])
    except Exception:
        return -1.0, -1.0

srv = subprocess.Popen([srv_exe, '-m', model, '--port', port] + extra,
                       stdout=open(out_f + '.server.log', 'w'), stderr=subprocess.STDOUT)
try:
    for _ in range(120):
        time.sleep(5)
        try:
            if req('/health', timeout=10).get('status') in ('ok', 'loading'):
                break
        except Exception:
            pass
    time.sleep(10)
    u0, t0 = gpu_idle()
    # warm-up (discarded)
    req('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Say hi.'}],
        'n_predict': 16, **samp}, timeout=300)
    rows = []
    for i, p in enumerate(prompts):
        if i:
            time.sleep(30)  # heat gap
        u, t = gpu_idle()
        d = req('/v1/chat/completions', {'messages': [{'role': 'user', 'content': p['text']}],
            'n_predict': 256, 'seed': 1000 + i, **samp})
        tm = d['timings']
        rows.append({'id': p['id'], 'prompt_n': tm['prompt_n'],
            'pp': round(tm['prompt_per_second'], 1), 'gen_n': tm['predicted_n'],
            'tg': round(tm['predicted_per_second'], 1),
            'draft_n': tm.get('draft_n', 0), 'accepted': tm.get('draft_n_accepted', 0),
            'acc': round(tm.get('draft_n_accepted', 0) / max(1, tm.get('draft_n', 0)), 3),
            'gpu_u': u, 'gpu_t': t})
        print(f"{p['id']}: pp={rows[-1]['pp']} tg={rows[-1]['tg']} acc={rows[-1]['acc']} gpu={u}%/{t}C",
              flush=True)
    json.dump({'idle_before': [u0, t0], 'rows': rows}, open(out_f, 'w'), indent=1)
    print('WROTE ' + out_f, flush=True)
finally:
    srv.terminate()
