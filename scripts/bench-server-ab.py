#!/usr/bin/env python3
"""Interleaved server A/B: alternate control/candidate in blocks, repeat prompts, pool acceptance.
Usage: bench-server-ab.py --ctl <server-exe> --cand <server-exe> --model <gguf>
    --port <p> --samp <sampling.json> --prompts <prompts.json> --out <out.json>
    [--order ABBABABA] [--repeats 3] [--n-predict 1024] [--device cuda|vulkan]
    [-- <extra server flags...>]
Prompts json: list of {id, text}. Same seed per prompt in every rep and block
(temp>0 + fixed seed => identical decode, so rep variance is pure system noise).
Acceptance is pooled over all reps/blocks per build: sum(accepted)/sum(draft_n).
"""
import json, subprocess, sys, time, urllib.request

a = sys.argv[1:]
o = {'order': 'ABBABABA', 'repeats': '3', 'n_predict': '1024', 'device': 'cuda'}
pos, i = [], 0
while i < len(a):
    if a[i].startswith('--') and i + 1 < len(a) and not a[i + 1].startswith('--') and a[i] != '--':
        o[a[i][2:]] = a[i + 1]; i += 2
    elif a[i] == '--':
        pos = a[i + 1:]; break
    else:
        pos.append(a[i]); i += 1
exe = {'A': o['ctl'], 'B': o['cand']}
model, port = o['model'], o['port']
samp = json.load(open(o['samp']))
prompts = json.load(open(o['prompts']))
REPS, NPRED, DEV = int(o['repeats']), int(o['n_predict']), o['device']
base = f'http://localhost:{port}'

def req(path, body=None, timeout=1200):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.urlopen(urllib.request.Request(base + path, data=data,
        headers={'Content-Type': 'application/json'}), timeout=timeout)
    return json.loads(r.read())

def idle():
    """Device-aware idle snapshot. CUDA: nvidia-smi util/temp. Vulkan(iGPU): CPU load, no nvidia-smi."""
    if DEV == 'cuda':
        try:
            s = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu',
                '--format=csv,noheader'], capture_output=True, text=True, timeout=30).stdout
            u, t = s.strip().split(',')
            return {'gpu_u': float(u.split()[0]), 'gpu_t': float(t.split()[0])}
        except Exception:
            return {'gpu_u': -1.0, 'gpu_t': -1.0}
    try:
        load = open('/proc/loadavg').read().split()[0:3]
        ncpu = subprocess.run(['nproc'], capture_output=True, text=True, timeout=10).stdout.strip()
        return {'cpu_load': [float(x) for x in load], 'cpu_n': int(ncpu)}
    except Exception:
        return {'cpu_load': [-1.0], 'cpu_n': -1}

def idle_ok(s):
    if DEV == 'cuda':
        return s.get('gpu_u', 99) < 5
    loads, n = s.get('cpu_load', [99]), s.get('cpu_n', 1)
    return loads[0] < max(1, n) * 0.5

def wait_idle(tag):
    for _ in range(120):
        s = idle()
        if idle_ok(s):
            return s
        time.sleep(5)
    print(f'WARN {tag}: box never idled, proceeding anyway', flush=True)
    return idle()

def serve_block(tag, build):
    srv = subprocess.Popen([exe[build], '-m', model, '--port', port] + pos,
                           stdout=open(f"{o['out']}.{tag}.server.log", 'w'),
                           stderr=subprocess.STDOUT)
    rows = []
    try:
        for _ in range(120):
            time.sleep(5)
            try:
                if req('/health', timeout=10).get('status') in ('ok', 'loading'):
                    break
            except Exception:
                pass
        time.sleep(10)
        wait_idle(f'{tag}-start')
        req('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Say hi.'}],
            'n_predict': 16, **samp}, timeout=300)  # warm-up, discarded
        for i, p in enumerate(prompts):
            for r in range(REPS):
                if i or r:
                    time.sleep(30)  # heat gap
                s = idle()
                d = req('/v1/chat/completions', {'messages': [{'role': 'user', 'content': p['text']}],
                    'n_predict': NPRED, 'seed': 1000 + i, **samp})
                tm = d['timings']
                rows.append({'block': tag, 'build': build, 'id': p['id'], 'rep': r,
                    'prompt_n': tm['prompt_n'], 'pp': round(tm['prompt_per_second'], 1),
                    'gen_n': tm['predicted_n'], 'tg': round(tm['predicted_per_second'], 1),
                    'draft_n': tm.get('draft_n', 0),
                    'accepted': tm.get('draft_n_accepted', 0), 'idle': s})
                print(f"{tag}/{build} {p['id']} r{r}: pp={rows[-1]['pp']} "
                      f"tg={rows[-1]['tg']} n={rows[-1]['gen_n']} idle={s}", flush=True)
    finally:
        srv.terminate()
    return rows

all_rows = []
for bi, b in enumerate(o['order']):
    wait_idle(f'block{bi}-{b}-pre')
    all_rows += serve_block(f'b{bi}', b)
    time.sleep(60)  # cooldown between blocks

summary = {}
for b in 'AB':
    br = [r for r in all_rows if r['build'] == b]
    dn = sum(r['draft_n'] for r in br)
    per = {}
    for p in prompts:
        pr = sorted(r['tg'] for r in br if r['id'] == p['id'])
        qr = sorted(r['pp'] for r in br if r['id'] == p['id'])
        per[p['id']] = {'pp_med': round(qr[len(qr) // 2], 1),
            'tg_med': round(pr[len(pr) // 2], 1),
            'tg_min': round(pr[0], 1), 'tg_max': round(pr[-1], 1)}
    summary[b] = {'n_runs': len(br),
        'pooled_acc': round(sum(r['accepted'] for r in br) / max(1, dn), 4),
        'draft_n_total': dn, 'per_prompt': per}
json.dump({'order': o['order'], 'repeats': REPS, 'n_predict': NPRED, 'device': DEV,
    'summary': summary, 'rows': all_rows}, open(o['out'], 'w'), indent=1)
print('WROTE ' + o['out'], flush=True)
for b in 'AB':
    print(f"build {b}: pooled_acc={summary[b]['pooled_acc']} "
          f"(n={summary[b]['draft_n_total']}) runs={summary[b]['n_runs']}", flush=True)
