# -*- coding: utf-8 -*-
"""
Profiling script for network.py Transformer model.
1. torch.profiler — operator-level time breakdown (CPU + CUDA)
2. Memory snapshot — per-operator GPU memory allocation tracking
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from network import Transformer


def profile_execution():
    """Use torch.profiler to trace operator execution times."""
    model = Transformer(num_of_layer=1, max_seq_len=8192).half().cuda()
    x = torch.randint(low=0, high=220000, size=[4455]).cuda()

    # Warmup
    for i in range(3):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    import datetime, json, os
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
    tb_log_dir = f"/data0/wjh528431/work/network/tb_log/{ts}"

    NOISE = ("profiler", "frozen importlib", "frozen zipimport", "importlib/",
             "torch/_inductor", "torch/utils/_config", "is_frozen",
             "PyCapsule", "bootstrap")

    def filtered_trace_handler(p):
        default_handler = torch.profiler.tensorboard_trace_handler(tb_log_dir)
        default_handler(p)
        for f in os.listdir(tb_log_dir):
            if not f.endswith(".json"):
                continue
            path = os.path.join(tb_log_dir, f)
            with open(path) as fh:
                data = json.load(fh)
            data["traceEvents"] = [
                ev for ev in data.get("traceEvents", [])
                if not any(n in ev.get("name", "") for n in NOISE)
            ]
            with open(path, "w") as fh:
                json.dump(data, fh)

    # --- add record_function markers so TensorBoard timeline shows sections ---
    active_rfs = []

    def _make_hooks(label):
        def pre_hook(mod, inp):
            rf = record_function(label)
            rf.__enter__()
            active_rfs.append(rf)
        def post_hook(mod, inp, out):
            if active_rfs:
                active_rfs.pop().__exit__(None, None, None)
        return pre_hook, post_hook

    hook_handles = []
    for name, module in model.named_modules():
        if name == "token_embedding":
            label = "embedding"
        elif name.endswith(".attn"):
            label = f"attention_{name.split('.')[1]}"
        elif name.endswith(".ffn"):
            label = f"ffn_{name.split('.')[1]}"
        elif name == "lm_head":
            label = "lm_head"
        else:
            continue
        pre, post = _make_hooks(label)
        hook_handles.append(module.register_forward_pre_hook(pre))
        hook_handles.append(module.register_forward_hook(post))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        schedule=torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1),
        on_trace_ready=filtered_trace_handler,
    ) as prof:
        prof.step()
        with record_function("model_forward"):
            y = model(x)
        prof.step()
    torch.cuda.synchronize()

    # cleanup hooks
    for h in hook_handles:
        h.remove()

    # Print time-sorted table
    print("=" * 100)
    print("OPERATOR TIME PROFILING (sorted by CUDA time)")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Print memory-sorted table
    print("\n" + "=" * 100)
    print("OPERATOR MEMORY PROFILING (sorted by GPU memory allocated)")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30))

    print(f"\nTensorBoard log exported to {tb_log_dir}")
    print("Run: tensorboard --logdir=" + tb_log_dir)


def profile_memory_per_operator():
    """Track GPU memory allocation per operator using hooks."""
    model = Transformer(num_of_layer=1, max_seq_len=8192).half().cuda()
    x = torch.randint(low=0, high=220000, size=[4455]).cuda()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    memory_records = []

    def make_hooks(name):
        def pre_hook(module, input):
            torch.cuda.synchronize()
            mem = torch.cuda.memory_allocated() / 1024**2
            memory_records.append({"name": name, "event": "before", "mem_MB": mem})

        def post_hook(module, input, output):
            torch.cuda.synchronize()
            mem = torch.cuda.memory_allocated() / 1024**2
            memory_records.append({"name": name, "event": "after", "mem_MB": mem})

        return pre_hook, post_hook

    # Register hooks on all named modules
    hooks = []
    for name, module in model.named_modules():
        if name == "":
            continue
        # Only hook leaf modules or key structural modules
        if len(list(module.children())) == 0 or name in [
            "token_embedding", "final_norm", "lm_head"
        ] or name.startswith("layers."):
            pre_hook, post_hook = make_hooks(name)
            hooks.append(module.register_forward_pre_hook(pre_hook))
            hooks.append(module.register_forward_hook(post_hook))

    # Run forward pass
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"\n{'=' * 100}")
    print("PER-OPERATOR GPU MEMORY USAGE")
    print(f"{'=' * 100}")
    print(f"Base memory (model weights + input): {base_mem:.2f} MB")
    print(f"{'=' * 100}")

    _ = model(x)
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak memory: {peak_mem:.2f} MB")
    print(f"Activation memory (peak - base): {peak_mem - base_mem:.2f} MB\n")

    # Compute per-operator memory delta
    print(f"{'Module':<60} {'Before(MB)':>12} {'After(MB)':>12} {'Delta(MB)':>12}")
    print("-" * 96)

    i = 0
    while i < len(memory_records) - 1:
        rec = memory_records[i]
        if rec["event"] == "before":
            # Find matching "after"
            for j in range(i + 1, len(memory_records)):
                if memory_records[j]["name"] == rec["name"] and memory_records[j]["event"] == "after":
                    before = rec["mem_MB"]
                    after = memory_records[j]["mem_MB"]
                    delta = after - before
                    if abs(delta) > 0.01:  # Only show non-trivial changes
                        print(f"{rec['name']:<60} {before:>12.2f} {after:>12.2f} {delta:>+12.2f}")
                    break
        i += 1

    # Remove hooks
    for h in hooks:
        h.remove()


def profile_memory_snapshot():
    """Use torch.cuda.memory._record_memory_history for detailed allocation tracking."""
    model = Transformer(num_of_layer=1, max_seq_len=8192).half().cuda()
    x = torch.randint(low=0, high=220000, size=[4455]).cuda()

    # Warmup
    _ = model(x)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Record memory history
    torch.cuda.memory._record_memory_history(max_entries=100000)

    _ = model(x)
    torch.cuda.synchronize()

    # Export snapshot
    torch.cuda.memory._dump_snapshot("/tmp/transformer_memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"\n{'=' * 100}")
    print("MEMORY SNAPSHOT")
    print(f"{'=' * 100}")
    print("Snapshot saved to /tmp/transformer_memory_snapshot.pickle")
    print("Visualize at: https://pytorch.org/memory_viz")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"  Current memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


if __name__ == "__main__":
    print(">>> Part 1: Operator Time Profiling")
    profile_execution()

    print("\n\n>>> Part 2: Per-Operator Memory Usage")
    profile_memory_per_operator()

    print("\n\n>>> Part 3: Memory Snapshot (for pytorch.org/memory_viz)")
    profile_memory_snapshot()
