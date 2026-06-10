# -*- coding: utf-8 -*-
"""Verify current network.py output matches initial_network.py (original implementation)."""
import torch
import sys

sys.path.insert(0, ".")
import initial_network
import network


if __name__ == "__main__":
    torch.manual_seed(42)

    # Build both models with same architecture, share weights
    original_model = initial_network.Transformer(num_of_layer=1, max_seq_len=8192).bfloat16().cuda()
    current_model = network.Transformer(num_of_layer=1, max_seq_len=8192).bfloat16().cuda()

    # Copy weights from original to current so they're identical
    current_model.load_state_dict(original_model.state_dict(), strict=False)

    x = torch.randint(low=0, high=220000, size=[4455]).cuda()

    with torch.no_grad():
        out_original = original_model(x)
        out_current = current_model(x)

    max_diff = (out_original - out_current).abs().max().item()
    mean_diff = (out_original - out_current).abs().mean().item()
    close = torch.allclose(out_original, out_current, atol=2e-2)

    print(f"Max abs diff:  {max_diff:.6e}")
    print(f"Mean abs diff: {mean_diff:.6e}")
    print(f"allclose (atol=2e-2): {close}")
    print(f"Result: {'PASS' if close else 'FAIL'}")

    del original_model, current_model, out_original, out_current, x
    torch.cuda.empty_cache()
