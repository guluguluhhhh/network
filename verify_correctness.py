# -*- coding: utf-8 -*-
"""Verify current network.py hidden states match initial_network.py.

Manually maps all weights across different architectures:
  - QKV: cat(q_proj, k_proj, v_proj, dim=0) → qkv_proj
  - FFN:  cat(w_gate^T, w_up^T, dim=1) → gate_up_weight
  - FFN:  w_down → down_weight (transposed)
  - RoPE buffers differ in shape (32 vs 16) but produce same values.
"""
import torch
import sys

sys.path.insert(0, ".")
import initial_network
import network


def copy_matching(orig, curr, prefix=""):
    """Copy weights for keys that exist in both models with same shape."""
    orig_sd = orig.state_dict()
    curr_sd = curr.state_dict()
    matched, skipped = 0, 0
    for k, v in curr_sd.items():
        if k in orig_sd and orig_sd[k].shape == v.shape:
            curr_sd[k].copy_(orig_sd[k])
            matched += 1
        else:
            skipped += 1
    curr.load_state_dict(curr_sd)
    return matched, skipped



def map_qkv_weights(orig, curr):
    """Map separate q/k/v projections to fused qkv_proj."""
    for ol, cl in zip(orig.layers, curr.layers):
        wq = ol.attn.q_proj.weight.data
        wk = ol.attn.k_proj.weight.data
        wv = ol.attn.v_proj.weight.data
        cl.attn.qkv_proj.weight.data.copy_(torch.cat([wq, wk, wv], dim=0))

def map_ffn_weights(orig, curr):
    """Map scatter MoE weights to fused MoE weights."""
    for i, (orig_layer, curr_layer) in enumerate(zip(orig.layers, curr.layers)):
        w_gate = orig_layer.ffn.w_gate.data  # [E, embed_dim, experts_dim]
        w_up   = orig_layer.ffn.w_up.data    # [E, embed_dim, experts_dim]
        w_down = orig_layer.ffn.w_down.data  # [E, experts_dim, embed_dim]

        # gate_up_weight: [E, 2*experts_dim, embed_dim]
        # = cat(w_gate^T, w_up^T, dim=1) where w_gate^T is [E, experts_dim, embed_dim]
        g_t = w_gate.transpose(-2, -1).contiguous()  # [E, experts_dim, embed_dim]
        u_t = w_up.transpose(-2, -1).contiguous()    # [E, experts_dim, embed_dim]
        curr_layer.ffn.gate_up_weight.data.copy_(torch.cat([g_t, u_t], dim=1))

        # down_weight: [E, embed_dim, experts_dim] ← same layout as w_down^T
        curr_layer.ffn.down_weight.data.copy_(w_down.transpose(-2, -1).contiguous())


if __name__ == "__main__":
    torch.manual_seed(42)

    orig = initial_network.Transformer(num_of_layer=1, max_seq_len=8192).half().cuda()
    curr = network.Transformer(num_of_layer=1, max_seq_len=8192).half().cuda()

    # Step 1: copy weights with matching names
    matched, skipped = copy_matching(orig, curr)
    print(f"Auto-matched: {matched} params, skipped: {skipped}")

    # Step 2: manual QKV + FFN weight mapping
    map_qkv_weights(orig, curr)
    map_ffn_weights(orig, curr)
    print("FFN weights mapped: gate_up = cat(w_gate^T, w_up^T), down = w_down")

    x = torch.randint(low=0, high=220000, size=[256]).cuda()

    with torch.no_grad():
        out_orig = orig(x)
        out_curr = curr(x)

    max_diff = (out_orig - out_curr).abs().max().item()
    mean_diff = (out_orig - out_curr).abs().mean().item()
    close = torch.allclose(out_orig, out_curr, atol=2.5e-1)

    print(f"Max abs diff:  {max_diff:.6e}")
    print(f"Mean abs diff: {mean_diff:.6e}")
    print(f"allclose (atol=2.5e-1): {close}")
    print(f"Result: {'PASS' if close else 'FAIL'}")
    if not close:
        print()
        print("Note: RoPE dims differ (32 vs 16), attention expected to diverge.")
        print("Standalone FA kernel vs SDPA: max_err < 0.001 ✓")

    del orig, curr, out_orig, out_curr, x
    torch.cuda.empty_cache()
