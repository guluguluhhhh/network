# -*- coding: utf-8 -*-
"""
PID Scheduler Benchmark with Paged KV Cache.

Realistic workload:
  - Prompt lengths: LogNormal (median ~50, range [16, 300])
  - Generation lengths: LogNormal HIGH VARIANCE (median ~50, long tail to 450)
  - Arrival: Poisson base + periodic 10x burst spikes

Shows PID controlling active_seqs under unpredictable exit rates.
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from network import Transformer
from components import PIDScheduler


# ==============================================================================
# Workload
# ==============================================================================

def get_arrival_rate(step: int) -> float:
    """Poisson base + burst spikes every 150 steps."""
    if step < 100:
        base = 5.0   # warm-up
    else:
        base = 12.0  # sustained high load
    # Burst: 10x for 5 steps, every 150 steps during high-load
    if step >= 100 and (step % 150) < 5:
        return base * 10.0
    return base


def generate_workload(num_steps: int, seed: int = 42):
    """Pre-generate: list of [(prompt_len, gen_len), ...] per step."""
    rng = np.random.RandomState(seed)
    workload = []
    for step in range(num_steps):
        rate = get_arrival_rate(step)
        num_arrivals = rng.poisson(rate)
        requests = []
        for _ in range(num_arrivals):
            plen = int(np.clip(rng.lognormal(4.0, 0.6), 16, 300))
            glen = int(np.clip(rng.lognormal(3.9, 1.0), 5, 450))
            requests.append((plen, glen))
        workload.append(requests)
    return workload


# ==============================================================================
# Benchmark
# ==============================================================================

@torch.inference_mode()
def run_single(workload, admit_mode="pid", num_steps=1000, label=""):
    """Run benchmark with given admit_mode. Returns metrics dict."""
    MAX_SEQ_LEN = 512
    NUM_PAGES = 2048
    MAX_BATCH = 512

    print(f"\n{'='*60}")
    print(f"  Mode: {admit_mode.upper()} {label}")
    print(f"{'='*60}")

    model = Transformer(num_of_layer=1, max_seq_len=MAX_SEQ_LEN).half().cuda()
    scheduler = PIDScheduler(
        model,
        max_batch_size=MAX_BATCH,
        target_ratio=0.8,
        kp=0.15, ki=0.005, kd=0.1,
        temperature=0.0,
        admit_mode=admit_mode,
        num_pages=NUM_PAGES,
    )

    target_pages = scheduler.pid.setpoint
    print(f"  num_pages={NUM_PAGES}, page_size={scheduler.kv_cache.page_size}, "
          f"target_pages={target_pages:.0f} ({target_pages/NUM_PAGES*100:.0f}%)")

    # Metrics
    steps, kv_usage, active_counts, queue_lens, pages_used = [], [], [], [], []
    total_submitted = 0
    total_completed = 0

    print(f"  Running {num_steps} steps...")
    for step in range(num_steps):
        for (prompt_len, gen_len) in workload[step]:
            prompt = torch.randint(0, 1000, [prompt_len], device='cuda')
            scheduler.add_request(prompt, max_gen_tokens=gen_len)
            total_submitted += 1

        completed = scheduler.step()
        total_completed += len(completed)

        steps.append(step)
        kv_usage.append(scheduler.kv_cache.total_kv_tokens)
        active_counts.append(len(scheduler.active_pool))
        queue_lens.append(len(scheduler.waiting_queue))
        pages_used.append(NUM_PAGES - scheduler.kv_cache.num_free)

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}: active={len(scheduler.active_pool)}, "
                  f"KV={scheduler.kv_cache.total_kv_tokens}, "
                  f"pages={NUM_PAGES - scheduler.kv_cache.num_free}/{NUM_PAGES}, "
                  f"queue={len(scheduler.waiting_queue)}")

    print(f"  Done: submitted={total_submitted}, completed={total_completed}")

    # Cleanup GPU memory for next run
    del scheduler, model
    torch.cuda.empty_cache()

    return dict(steps=steps, kv_usage=kv_usage, active_counts=active_counts,
                queue_lens=queue_lens, pages_used=pages_used,
                target_pages=target_pages, num_pages=NUM_PAGES)


def run_benchmark(num_steps=1000):
    """Run both PID and No-PID on identical workload, return comparison."""
    print("Generating workload...")
    workload = generate_workload(num_steps)
    total_reqs = sum(len(w) for w in workload)
    gen_lens = [g for step in workload for (_, g) in step]
    print(f"  Requests: {total_reqs}, gen_len median={np.median(gen_lens):.0f}, "
          f"mean={np.mean(gen_lens):.0f}, p99={np.percentile(gen_lens, 99):.0f}")

    pid_results = run_single(workload, admit_mode="pid", num_steps=num_steps,
                             label="(controlled)")
    nopid_results = run_single(workload, admit_mode="nopid", num_steps=num_steps,
                               label="(all-admit)")
    return pid_results, nopid_results


# ==============================================================================
# Plot (PID vs No-PID comparison)
# ==============================================================================

def plot_results(pid, nopid):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("PID vs No-PID Scheduler Comparison\n"
                 "(identical workload: LogNormal prompt/gen, burst arrivals)",
                 fontsize=13, fontweight='bold')

    target_pages = pid['target_pages']
    num_pages = pid['num_pages']

    # -- Top: Active Sequences --
    ax1.plot(pid['steps'], pid['active_counts'], 'b-', linewidth=1.0,
             label='PID (active)', alpha=0.9)
    ax1.plot(nopid['steps'], nopid['active_counts'], 'r-', linewidth=1.0,
             label='No-PID (active)', alpha=0.7)
    ax1.set_ylabel('Active Sequences')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Active Sequences')
    for s in range(100, 1000, 150):
        ax1.axvspan(s, s + 5, alpha=0.06, color='purple')

    # -- Bottom: Pages Used (KV pressure) --
    ax2.plot(pid['steps'], pid['pages_used'], 'b-', linewidth=1.0,
             label='PID (pages)', alpha=0.9)
    ax2.plot(nopid['steps'], nopid['pages_used'], 'r-', linewidth=1.0,
             label='No-PID (pages)', alpha=0.7)
    ax2.axhline(y=num_pages, color='gray', linestyle=':', linewidth=1,
                label=f'Capacity ({num_pages})')
    ax2.axhline(y=target_pages, color='green', linestyle='--', linewidth=1.5,
                label=f'PID target ({int(target_pages)}, {target_pages/num_pages*100:.0f}%)')
    ax2.set_ylabel('Pages Used')
    ax2.set_xlabel('Step')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('KV Page Usage: PID controls towards 80% target')
    for s in range(100, 1000, 150):
        ax2.axvspan(s, s + 5, alpha=0.06, color='purple')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "pid_vs_nopid_bench.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.close()

    # Print summary comparison
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Metric':<25} {'PID':>12} {'No-PID':>12} {'Diff':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    pa = np.max(pid['pages_used'])
    na = np.max(nopid['pages_used'])
    print(f"  {'Peak pages used':<25} {pa:>12} {na:>12} {na-pa:>+10}")
    pa2 = np.mean(pid['active_counts'])
    na2 = np.mean(nopid['active_counts'])
    print(f"  {'Mean active seqs':<25} {pa2:>12.1f} {na2:>12.1f} {na2-pa2:>+10.1f}")
    print("="*60)


if __name__ == "__main__":
    pid_results, nopid_results = run_benchmark(num_steps=1000)
    plot_results(pid_results, nopid_results)
