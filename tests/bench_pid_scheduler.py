"""
PID Scheduler Benchmark: Real model inference with pool+slot_mapping KV cache.

Client: Poisson-distributed request arrivals with load phases.
Server: Real PIDScheduler + Transformer (actual GPU forward pass each step).
Output: PNG plot showing KV cache utilization and request latency.
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


def get_arrival_rate(step: int) -> float:
    if step < 200:
        return 30.0
    elif step < 600:
        return 80.0
    else:
        return 35.0


def sample_prompt_len() -> int:
    return int(np.clip(np.random.normal(32, 8), 16, 64))


@torch.inference_mode()
def run_benchmark(num_steps=1000):
    print("Initializing model + PID scheduler...")
    model = Transformer(num_of_layer=1, max_seq_len=128).half().cuda()
    # pool = 4000 slots * 64KB/slot = 256MB, target_active = 0.8*4000 = 3200
    scheduler = PIDScheduler(
        model,
        max_batch_size=4000,
        target_ratio=0.8,
        kp=0.5, ki=0.01, kd=0.3,
        temperature=0.0,
    )

    max_total_tokens = scheduler.max_total_tokens
    target_active = scheduler.pid.setpoint
    print(f"  max_slots={scheduler.kv_cache.max_slots}, target_active={target_active:.0f} seqs")
    print(f"  max_total_tokens={max_total_tokens}")

    # Metrics
    steps = []
    kv_usage = []
    active_counts = []
    queue_lens = []
    arrival_rates = []

    submit_times = {}
    latencies = []

    np.random.seed(42)
    total_submitted = 0
    total_completed = 0

    print(f"\nRunning {num_steps} steps...")
    for step in range(num_steps):
        # Client: inject requests
        rate = get_arrival_rate(step)
        num_arrivals = np.random.poisson(rate)
        for _ in range(num_arrivals):
            prompt_len = sample_prompt_len()
            prompt = torch.randint(0, 1000, [prompt_len], device='cuda')
            req_id = scheduler.add_request(prompt)
            submit_times[req_id] = step
            total_submitted += 1

        # Server: one real forward step
        completed = scheduler.step()

        for req in completed:
            submit_step = submit_times.pop(req.id, step)
            latencies.append((step, step - submit_step, req.prompt_ids.size(0)))
            total_completed += 1

        # Metrics
        kv_used = scheduler.kv_cache.total_kv_tokens
        steps.append(step)
        kv_usage.append(kv_used)
        active_counts.append(len(scheduler.active_pool))
        queue_lens.append(len(scheduler.waiting_queue))
        arrival_rates.append(rate)

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}: KV={kv_used}, active={len(scheduler.active_pool)}, "
                  f"queue={len(scheduler.waiting_queue)}, completed={total_completed}")

    # Drain
    drain_steps = 0
    while scheduler.has_work() and drain_steps < 300:
        completed = scheduler.step()
        for req in completed:
            submit_step = submit_times.pop(req.id, num_steps + drain_steps)
            latencies.append((num_steps + drain_steps, (num_steps + drain_steps) - submit_step, req.prompt_ids.size(0)))
            total_completed += 1
        steps.append(num_steps + drain_steps)
        kv_usage.append(scheduler.kv_cache.total_kv_tokens)
        active_counts.append(len(scheduler.active_pool))
        queue_lens.append(len(scheduler.waiting_queue))
        arrival_rates.append(0.0)
        drain_steps += 1

    print(f"\nDone: submitted={total_submitted}, completed={total_completed}, drain={drain_steps}")
    if latencies:
        lats = [l[1] for l in latencies]
        print(f"Latency: mean={np.mean(lats):.1f}, p50={np.median(lats):.1f}, p99={np.percentile(lats, 99):.1f}")

    return steps, kv_usage, active_counts, queue_lens, arrival_rates, latencies, target_active, max_total_tokens


def plot_results(steps, kv_usage, active_counts, queue_lens, arrival_rates,
                 latencies, target_active, max_total_tokens):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.suptitle("PID Continuous Batching Scheduler (pool+slot_mapping)", fontsize=13)

    # Top: active seqs + KV
    ax1.plot(steps, active_counts, 'b-', linewidth=1, label='Active seqs')
    ax1.axhline(y=target_active, color='r', linestyle='--', linewidth=1.5,
                label=f'PID target ({target_active:.0f} seqs)')
    ax1.set_ylabel('Active Sequences', color='b')
    ax1.set_xlabel('Step')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Active Sequences (PID-controlled) & Queue Length')

    ax1.axvspan(0, 200, alpha=0.05, color='blue')
    ax1.axvspan(200, 600, alpha=0.05, color='red')
    ax1.axvspan(600, 1000, alpha=0.05, color='green')

    ax1b = ax1.twinx()
    ax1b.plot(steps, queue_lens, 'orange', linewidth=0.8, alpha=0.7, label='Queue')
    ax1b.set_ylabel('Queue Length', color='orange')

    # Bottom: latency
    if latencies:
        cs = [l[0] for l in latencies]
        ls = [l[1] for l in latencies]
        colors = ['blue' if s < 200 else 'red' if s < 600 else 'green' for s in cs]
        ax2.scatter(cs, ls, c=colors, s=6, alpha=0.5)
        ax2.axhline(y=np.mean(ls), color='black', linestyle='--', linewidth=1,
                    label=f'Mean={np.mean(ls):.0f}')
        ax2.set_xlabel('Completion Step')
        ax2.set_ylabel('Latency (steps)')
        ax2.set_title('Per-Request Latency')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "pid_bench_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    results = run_benchmark(num_steps=1000)
    plot_results(*results)
