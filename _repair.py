"""One-shot repair of the mangled VAIP definitions in the public repo scripts."""
import re

BS = chr(92)
VAIP_LINE = ('VAIP = os.environ.get("VAIP_CONFIG", r"C:' + BS + 'Program Files' + BS +
             'RyzenAI' + BS + '1.8.0' + BS + 'voe-4.0-win_amd64' + BS +
             'vaip_config.json")')

FILES = [
    'verify/npu_bench20.py',
    'verify/e2e_npu_vision.py',
    'verify/npu_run.py',
    'verify/npu_vision3.py',
    'verify/npu_op_probe.py',
    'try/try_d1_npu_stage.py',
]

for f in FILES:
    s = open(f, encoding='utf-8').read()
    lines = [l for l in s.split('\n')]

    out, inserted = [], 'VAIP' in s
    for l in lines:
        if 'rVAIP' in l:
            continue  # broken remnant line
        if re.match(r'\s*VAIP[_A-Z]*\s*=', l) or 'VAIP_CONFIG' in l and '=' in l and 'os.environ' not in l:
            continue  # drop any existing (possibly mangled) VAIP/VAIP_CONFIG def; re-added below
        if 'RyzenAI.8.0' in l or 'Program FilesRyzenAI' in l:
            continue
        out.append(l)

    s = '\n'.join(out)
    if 'VAIP =' not in s:
        lines = s.split('\n')
        imp_idx = max(i for i, l in enumerate(lines) if l.startswith('import ') or l.startswith('from '))
        lines.insert(imp_idx + 1, VAIP_LINE)
        s = '\n'.join(lines)
    while 'import os\nimport os' in s:
        s = s.replace('import os\nimport os', 'import os')
    if 'import os\n' not in s and 'import os' not in s.split('\n\n')[0]:
        s = 'import os\n' + s
    compile(s, f, 'exec')
    open(f, 'w', encoding='utf-8', newline='').write(s)
    print('repaired', f)

print('---')
for f in FILES:
    n = sum(1 for l in open(f, encoding='utf-8') if 'rVAIP' in l)
    print(f, 'rVAIP occurrences:', n)