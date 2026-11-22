"""Diagnostic: replay one n5 hold episode (relays still) and print PDR per 30 s window.

Run from the repo root in Linux: python3 scripts/analysis/arp_blackout_repro.py LAYOUT RADIO [extra simulator flags]
e.g. 2800063 2900063 reproduces test-n5-gradual-03 (PDR 0.497); add --ns3::ArpCache::DeadTimeout=+1s to remove
the 100 s startup ARP blackout (0.955). Not part of any pre-registered analysis."""
import json, subprocess, sys, tempfile, uuid
layout, radio, *extra = sys.argv[1:]
out = tempfile.mkdtemp(prefix='v5diag-')
cmd = ['build/recovery/uav-olsr-dataset', '--enableServiceObservation=true', '--scenarioMode=relay-prevention',
       '--recoveryPolicy=external', '--numUavs=5', '--timeSteps=241', f'--seed={radio}', f'--recoveryLayoutSeed={layout}',
       '--recoveryRandomLayout=true', '--sourceId=0', '--destId=4', '--warmup=10.0', '--xMax=1800.0', '--yMax=1200.0',
       '--zMin=50.0', '--zMax=150.0', '--commRange=240.0', '--recoverySpeed=8.0', '--recoveryBudget=400.0',
       '--recoveryMinSeparation=5.0', '--recoveryConfirmWindows=3', '--preventionMissionSpeed=1.0',
       '--preventionMissionStartStep=30', '--preventionSpacingFraction=0.1', '--routingStrategy=olsr',
       f'--preventionOutputReservation={uuid.uuid4().hex}', '--enableDataFlow=true', '--enableAnim=false',
       f'--outputDir={out}/ep', '--routeControl=external'] + extra
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(out + "/stderr.log", "w"), text=True, bufsize=1)
win = {}
for line in p.stdout:
    if not line.startswith('UAVRL '): continue
    f = json.loads(line[6:]); s = f.get('service_observation')
    if s:
        b = win.setdefault(s['sim_time_s'] // 30 * 30, [0, 0]); b[0] += s['window']['offered_packets']; b[1] += s['window']['sink_rx_packets']
    if f['step'] >= 240:
        p.stdin.write('UAVQUIT\n'); p.stdin.flush(); break
    p.stdin.write(f"UAVACT 2 {f['step']} " + ' '.join(['0'] * 9) + ' ROUTE 0\n'); p.stdin.flush()
p.wait(timeout=60)
off = sum(b[0] for b in win.values()); rx = sum(b[1] for b in win.values())
print(' '.join(extra) or 'default', f'pdr={rx/off:.3f}', ' '.join(f'{k}:{v[1]/v[0]:.2f}' for k, v in sorted(win.items()) if v[0]))
