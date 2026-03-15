import json
with open('results/mirai3/metrics.json') as f:
    m = json.load(f)
for k, v in m.items():
    print(f'{k}: {v}')