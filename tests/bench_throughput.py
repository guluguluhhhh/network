# -*- coding: utf-8 -*-
"""
Throughput Benchmark (vLLM-style).

Measures wall-clock throughput of the inference engine:
  - Request throughput: completed requests / second
  - Output throughput: generated tokens / second
  - Total token throughput: (input + output) tokens / second

Client sends requests following Poisson arrival at configurable QPS.
Engine uses autonomous stop (LogNormal hazard rate, IETF standard).
"""

import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from network import Transformer
from components import PIDScheduler


# ==============================================================================
# Client: Request Generator (vLLM-style Poisson arrival)
# ==============================================================================

class BenchClient:
    """Generates requests at target QPS with Poisson arrival."""

    def __init__(self, target_qps: float, prompt_mu: float = 4.0,
                 prompt_sigma: float = 0.6, seed: int = 42):
        self.target_qps = target_qps
        self.prompt_mu = prompt_mu
        self.prompt_sigma = prompt_sigma
        self.rng = np.random.RandomState(seed)

    def generate_batch(self, duration_steps: int) -> list[list[int]]:
        """Pre-generate workload: list of [prompt_len, ...] per step.

        Each step, num_arrivals ~ Poisson(target_qps).
        prompt_len ~ LogNormal(prompt_mu, prompt_sigma), clipped [16, 512].
        """
        workload = []
        for _ in range(duration_steps):
            n = self.rng.poisson(self.target_qps)
            prompts = []
            for _ in range(n):
                plen = int(np.clip(
                    self.rng.lognormal(self.prompt_mu, self.prompt_sigma), 16, 512))
                prompts.append(plen)
            workload.append(prompts)
        return workload


# ==============================================================================
# Benchmark Runner
# ==============================================================================

@torch.inference_mode()
def run_throughput_bench(
    target_qps: float = 25.0,
    num_steps: int = 500,
    num_pages: int = 2048,
    max_batch: int = 512,
    admit_mode: str = "pid",
):
    """Run throughput benchmark. Returns metrics dict."""
    MAX_SEQ_LEN = 512

    print(f"\n{'='*60}")
    print(f"  Throughput Benchmark")
    print(f"  Mode: {admit_mode.upper()}, QPS: {target_qps}, Steps: {num_steps}")
    print(f"{'='*60}")

    # Setup
    model = Transformer(num_of_layer=1, max_seq_len=MAX_SEQ_LEN).half().cuda()
    scheduler = PIDScheduler(
        model,
        max_batch_size=max_batch,
        target_ratio=0.8,
        kp=0.4, ki=0.02, kd=0.15,
        temperature=0.0,
        admit_mode=admit_mode,
        num_pages=num_pages,
    )

    client = BenchClient(target_qps=target_qps)
    workload = client.generate_batch(num_steps)
    total_requests = sum(len(w) for w in workload)
    total_input_tokens = sum(p for step in workload for p in step)

    print(f"  Total requests: {total_requests}")
    print(f"  Total input tokens: {total_input_tokens}")
    print(f"  Engine stop: LogNormal(mu=4.5, sigma=1.2), median~90 tokens")
    print()

    # Run
    total_completed = 0
    total_output_tokens = 0
    total_preempted = 0
    step_times = []

    # Per-request timing: track submit_step and first_decode_step for TTFT
    submit_step_map = {}    # req_id -> step when submitted
    first_decode_step = {}  # req_id -> step when first decode (prefill done)
    ttfts = []              # TTFT in steps (first token latency)
    tpots = []              # TPOT in ms (time per output token)
    e2els = []              # E2E latency in steps

    print(f"  Running {num_steps} steps...")
    bench_start = time.perf_counter()

    for step in range(num_steps):
        step_start = time.perf_counter()

        # Inject requests
        for prompt_len in workload[step]:
            prompt = torch.randint(0, 1000, [prompt_len], device='cuda')
            req_id = scheduler.add_request(prompt)
            submit_step_map[req_id] = step

        # Track which requests transition from prefill to decode this step
        for seq_id, req in scheduler.active_pool.items():
            if req.is_prefilled and seq_id not in first_decode_step:
                first_decode_step[seq_id] = step

        # Execute one scheduling iteration
        completed = scheduler.step()

        step_end = time.perf_counter()
        step_ms = (step_end - step_start) * 1000
        step_times.append(step_ms)

        # Track newly prefilled requests (just became is_prefilled this step)
        for seq_id, req in scheduler.active_pool.items():
            if req.is_prefilled and seq_id not in first_decode_step:
                first_decode_step[seq_id] = step

        # Count output tokens from completed requests + compute latencies
        for req in completed:
            gen_len = len(req.generated_ids)
            total_output_tokens += gen_len

            # TTFT: steps from submit to first token
            sub_step = submit_step_map.pop(req.id, step)
            ft_step = first_decode_step.pop(req.id, sub_step + 1)
            ttft = ft_step - sub_step
            ttfts.append(ttft)

            # E2E latency: steps from submit to completion
            e2e = step - sub_step
            e2els.append(e2e)

            # TPOT: step_ms / 1 (each decode step produces 1 token)
            # Use average step time as proxy for per-token time
            if gen_len > 1:
                tpots.append(step_ms)  # last step's latency as TPOT sample

        total_completed += len(completed)

        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - bench_start
            print(f"  Step {step+1}/{num_steps}: "
                  f"completed={total_completed}, "
                  f"active={len(scheduler.active_pool)}, "
                  f"pages={num_pages - scheduler.kv_cache.num_free}/{num_pages}, "
                  f"elapsed={elapsed:.1f}s")

    bench_end = time.perf_counter()
    total_preempted = getattr(scheduler, '_preempted_count', 0)

    # Metrics
    elapsed_s = bench_end - bench_start

    req_throughput = total_completed / elapsed_s
    output_throughput = total_output_tokens / elapsed_s
    total_throughput = (total_input_tokens + total_output_tokens) / elapsed_s
    mean_step_ms = np.mean(step_times)
    p99_step_ms = np.percentile(step_times, 99)

    # TTFT/TPOT stats (in wall-clock ms)
    # TTFT = ttft_steps * mean_step_ms (convert steps to ms)
    ttft_ms = [t * mean_step_ms for t in ttfts] if ttfts else [0]
    # TPOT = mean step time when batch is in decode-dominant state
    tpot_ms = tpots if tpots else [mean_step_ms]
    e2el_ms = [e * mean_step_ms for e in e2els] if e2els else [0]

    print(f"\n{'='*60}")
    print(f"  RESULTS ({admit_mode.upper()})")
    print(f"{'='*60}")
    print(f"  Duration:              {elapsed_s:.2f} s")
    print(f"  Submitted:             {total_requests}")
    print(f"  Completed:             {total_completed}")
    print(f"  Preempted (failed):    {total_preempted}")
    print(f"  Failure rate:          {total_preempted/max(total_requests,1)*100:.2f}%")
    print(f"  ---")
    print(f"  Request throughput:    {req_throughput:.1f} req/s")
    print(f"  Output throughput:     {output_throughput:.0f} tok/s")
    print(f"  Total throughput:      {total_throughput:.0f} tok/s (in+out)")
    print(f"  ---")
    print(f"  TTFT mean:             {np.mean(ttft_ms):.1f} ms")
    print(f"  TTFT p99:              {np.percentile(ttft_ms, 99):.1f} ms")
    print(f"  TPOT mean:             {np.mean(tpot_ms):.1f} ms")
    print(f"  TPOT p99:              {np.percentile(tpot_ms, 99):.1f} ms")
    print(f"  E2EL mean:             {np.mean(e2el_ms):.1f} ms")
    print(f"  E2EL p99:              {np.percentile(e2el_ms, 99):.1f} ms")
    print(f"  ---")
    print(f"  Step latency mean:     {mean_step_ms:.1f} ms")
    print(f"  Step latency p99:      {p99_step_ms:.1f} ms")
    print(f"{'='*60}")

    del scheduler, model
    torch.cuda.empty_cache()

    return dict(
        elapsed_s=elapsed_s,
        submitted=total_requests,
        completed=total_completed,
        preempted=total_preempted,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        req_throughput=req_throughput,
        output_throughput=output_throughput,
        total_throughput=total_throughput,
        mean_step_ms=mean_step_ms,
        p99_step_ms=p99_step_ms,
        ttft_mean_ms=np.mean(ttft_ms),
        ttft_p99_ms=np.percentile(ttft_ms, 99),
        tpot_mean_ms=np.mean(tpot_ms),
        tpot_p99_ms=np.percentile(tpot_ms, 99),
        e2el_mean_ms=np.mean(e2el_ms),
        e2el_p99_ms=np.percentile(e2el_ms, 99),
    )


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  vLLM-style Throughput Benchmark")
    print("  Poisson arrival + autonomous stop + PID admission control")
    print("=" * 60)

    pid_metrics = run_throughput_bench(admit_mode="pid", target_qps=25, num_steps=500)
    nopid_metrics = run_throughput_bench(admit_mode="nopid", target_qps=25, num_steps=500)

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<28} {'PID':>15} {'No-PID':>15}")
    print(f"  {'-'*28} {'-'*15} {'-'*15}")
    print(f"  {'Req throughput (req/s)':<28} {pid_metrics['req_throughput']:>15.1f} {nopid_metrics['req_throughput']:>15.1f}")
    print(f"  {'Output throughput (tok/s)':<28} {pid_metrics['output_throughput']:>15.0f} {nopid_metrics['output_throughput']:>15.0f}")
    print(f"  {'Total throughput (tok/s)':<28} {pid_metrics['total_throughput']:>15.0f} {nopid_metrics['total_throughput']:>15.0f}")
    print(f"  {'Failure rate':<28} {pid_metrics['preempted']/max(pid_metrics['submitted'],1)*100:>14.2f}% {nopid_metrics['preempted']/max(nopid_metrics['submitted'],1)*100:>14.2f}%")
    print(f"  {'TTFT mean (ms)':<28} {pid_metrics['ttft_mean_ms']:>15.1f} {nopid_metrics['ttft_mean_ms']:>15.1f}")
    print(f"  {'TTFT p99 (ms)':<28} {pid_metrics['ttft_p99_ms']:>15.1f} {nopid_metrics['ttft_p99_ms']:>15.1f}")
    print(f"  {'TPOT mean (ms)':<28} {pid_metrics['tpot_mean_ms']:>15.1f} {nopid_metrics['tpot_mean_ms']:>15.1f}")
    print(f"  {'TPOT p99 (ms)':<28} {pid_metrics['tpot_p99_ms']:>15.1f} {nopid_metrics['tpot_p99_ms']:>15.1f}")
    print(f"  {'E2EL mean (ms)':<28} {pid_metrics['e2el_mean_ms']:>15.1f} {nopid_metrics['e2el_mean_ms']:>15.1f}")
    print(f"  {'E2EL p99 (ms)':<28} {pid_metrics['e2el_p99_ms']:>15.1f} {nopid_metrics['e2el_p99_ms']:>15.1f}")
    print(f"  {'Step latency mean (ms)':<28} {pid_metrics['mean_step_ms']:>15.1f} {nopid_metrics['mean_step_ms']:>15.1f}")
    print(f"  {'Step latency p99 (ms)':<28} {pid_metrics['p99_step_ms']:>15.1f} {nopid_metrics['p99_step_ms']:>15.1f}")
    print(f"{'='*60}")
