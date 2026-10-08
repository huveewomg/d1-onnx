import sys, time, psutil, statistics
dur=float(sys.argv[1]); tag=sys.argv[2]
samples=[]; t0=time.time()
while time.time()-t0 < dur:
    samples.append(psutil.cpu_percent(interval=0.5))
print(f"{tag}: CPU%% mean={statistics.mean(samples):.0f} p50={statistics.median(samples):.0f} n={len(samples)}")
