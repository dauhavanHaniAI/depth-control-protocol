import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F



# CONFIG
@dataclass
class VERAv6Args:
    # ── Backbone ──
    dim: int = 1536
    n_heads: int = 24
    n_kv_heads: int = 4
    vocab_size: int = 32000

    n_perception_layers: int = 16
    n_reasoning_blocks: int = 2
    max_reasoning_iters: int = 4
    n_thought_tokens: int = 32

    ffn_hidden: int = 4352
    norm_eps: float = 1e-6
    max_seq_len: int = 2048
    dropout: float = 0.0

    init_scale_with_depth: bool = True
    gradient_checkpointing: bool = False

    # ── DSP: Dual-Stream RoPE (from v5, simplified) ──
    rope_thetas_lang: List[float] = field(
        default_factory=lambda: [500.0, 10_000.0, 500_000.0]
    )
    rope_thetas_op: List[float] = field(
        default_factory=lambda: [50.0, 500.0, 5_000.0]
    )
    n_streams: int = 2

    # ── TTM: Tree Thought Memory (from v5, fixed) ──
    n_branches: int = 2
    branch_router_hidden: int = 256
    use_tree_memory: bool = True

    # ── IVH: Inline Verifier (from v5) ──
    use_verifier: bool = True
    verifier_hidden: int = 384
    verifier_loss_weight: float = 0.3
    deep_supervision: bool = True

    # ── EHG: Energy Halt Gate (from v5, FIXED) ──
    halt_mode: str = "energy"       # "sigmoid" | "energy"
    energy_temperature: float = 1.0
    energy_reg_weight: float = 0.01

    # ── NEW: MELT Gated KV Cache ──
    use_melt_kv: bool = True        # gated latent state for reasoning KV

    # ── Causal Thought Bridge (v6.2) ──
    # Restores original intent: thought path is INPUT-DEPENDENT while remaining
    # causal (no future-token leak). See CausalThoughtBridge.
    #   "causal_cross" — recommended (prefix-mean read + pos-wise write)
    #   "independent"  — legacy over-fix (thought ignores real; verifier/TTM dead)
    thought_read_mode: str = "causal_cross"
    use_causal_thought_bridge: bool = True

    # ── ★ NEW: Lighthouse Attention ──
    # v6.1: default OFF. With top_k=1536 ≈ seq_len=2048 it gives near-zero
    # savings, the gather/scatter adds overhead causing OOM on 40GB GPUs,
    # and the causal mask on the scattered subsequence is not correctly
    # ordered (information leak). Enable only when top_k << seq_len.
    use_lighthouse: bool = False
    lighthouse_levels: int = 3      # L pyramid levels
    lighthouse_pool_factor: int = 2 # p pooling factor
    lighthouse_top_k: int = 512     # k budget per head (must be << seq_len)
    # Skip first/last N perception layers (keep dense for stability)
    lighthouse_skip_first: int = 2
    lighthouse_skip_last: int = 2


# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x.float() * norm).type_as(x)


# DSP — Dual-Stream RoPE
class DualStreamRoPE(nn.Module):
    """
    Per-token stream-aware multi-scale RoPE.

    v6 simplification vs v5: instead of 6D gather for per-token dispatch,
    compute BOTH stream freqs, then use stream_ids as binary mask to blend.
    Same result, much simpler + faster.
    """
    def __init__(self, head_dim: int, n_heads: int, args: VERAv6Args):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_scales = len(args.rope_thetas_lang)
        self.n_streams = args.n_streams

        max_len = args.max_seq_len + args.n_thought_tokens

        for s, theta in enumerate(args.rope_thetas_lang):
            self.register_buffer(f"freqs_lang_{s}",
                                 self._build_freqs(head_dim, max_len, theta),
                                 persistent=False)
        for s, theta in enumerate(args.rope_thetas_op):
            self.register_buffer(f"freqs_op_{s}",
                                 self._build_freqs(head_dim, max_len, theta),
                                 persistent=False)

        # [n_streams, n_heads, n_scales]
        self.mix_logits = nn.Parameter(
            torch.zeros(self.n_streams, n_heads, self.n_scales)
        )

    @staticmethod
    def _build_freqs(head_dim, seq_len, theta):
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        m = torch.arange(seq_len).float()
        freqs = torch.outer(m, inv)
        return torch.stack([freqs.cos(), freqs.sin()], dim=-1)

    def _blend_stream_freqs(self, n_h_local, stream_weights, start, seq_len):
        """
        Blend freqs for each stream, return [n_streams, n_h_local, seq, hd/2, 2].
        """
        results = []
        for s_idx, prefix in enumerate(["lang", "op"]):
            w = stream_weights[s_idx, :n_h_local]  # [n_h_local, n_scales]
            all_f = torch.stack([
                getattr(self, f"freqs_{prefix}_{sc}")[start:start + seq_len]
                for sc in range(self.n_scales)
            ], dim=0)  # [n_scales, seq, hd/2, 2]
            w_e = w.view(n_h_local, self.n_scales, 1, 1, 1)
            blended = (w_e * all_f.unsqueeze(0)).sum(dim=1)  # [n_h, seq, hd/2, 2]
            results.append(blended)
        return torch.stack(results, dim=0)  # [2, n_h, seq, hd/2, 2]

    def _rotate_with_stream_ids(self, x, per_stream_freqs, stream_ids):
        """
        x: [B, seq, n_h, hd]
        per_stream_freqs: [n_streams, n_h, seq, hd/2, 2]
        stream_ids: [B, seq] int in {0, 1}

        v6 FIX: instead of 6D gather, compute rotation for BOTH streams
        then select with stream_ids mask. Simpler + torch.compile friendly.
        """
        B, seq, n_h, hd = x.shape
        x_pairs = x.float().reshape(B, seq, n_h, hd // 2, 2)

        # Compute rotation for stream 0 and stream 1
        results = []
        for s in range(2):
            f = per_stream_freqs[s]  # [n_h, seq, hd/2, 2]
            cos_f = f[:, :, :, 0].permute(1, 0, 2).unsqueeze(0)  # [1, seq, n_h, hd/2]
            sin_f = f[:, :, :, 1].permute(1, 0, 2).unsqueeze(0)
            x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
            out0 = x0 * cos_f - x1 * sin_f
            out1 = x0 * sin_f + x1 * cos_f
            rotated = torch.stack([out0, out1], dim=-1).reshape(B, seq, n_h, hd)
            results.append(rotated)

        r0, r1 = results[0], results[1]  # [B, seq, n_h, hd]

        # Select based on stream_ids: [B, seq] → [B, seq, 1, 1]
        mask = stream_ids.unsqueeze(-1).unsqueeze(-1).float()  # 0 or 1
        return ((1.0 - mask) * r0 + mask * r1).type_as(x)

    def forward(self, q, k, start_pos, stream_ids):
        seq_len = q.shape[1]
        n_kv = k.shape[2]
        n_hq = q.shape[2]

        weights = F.softmax(self.mix_logits, dim=-1)

        q_freqs = self._blend_stream_freqs(n_hq, weights, start_pos, seq_len)
        q_out = self._rotate_with_stream_ids(q, q_freqs, stream_ids)

        n_rep = n_hq // n_kv
        kv_weights = weights[:, :n_hq].view(
            self.n_streams, n_kv, n_rep, self.n_scales
        ).mean(dim=2)
        k_freqs = self._blend_stream_freqs(n_kv, kv_weights, start_pos, seq_len)
        k_out = self._rotate_with_stream_ids(k, k_freqs, stream_ids)

        return q_out, k_out

# GQA Attention with DSP
def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, seq, n_kv, hd = x.shape
    return x[:, :, :, None, :].expand(B, seq, n_kv, n_rep, hd).reshape(
        B, seq, n_kv * n_rep, hd)


class DSPAttention(nn.Module):
    def __init__(self, args: VERAv6Args):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.rope = DualStreamRoPE(self.head_dim, args.n_heads, args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, stream_ids, start_pos=0, causal_mask=None,
                k_override=None, v_override=None):
        """
        k_override / v_override: pre-computed [B, seq, n_kv_heads*head_dim]
        tensors that bypass self.wk / self.wv. Used by MELT to route K, V
        through the gated latent state instead of recomputing from x.
        """
        B, seq, _ = x.shape
        xq = self.wq(x).view(B, seq, self.n_heads, self.head_dim)
        if k_override is None:
            xk = self.wk(x).view(B, seq, self.n_kv_heads, self.head_dim)
        else:
            xk = k_override.view(B, seq, self.n_kv_heads, self.head_dim)
        if v_override is None:
            xv = self.wv(x).view(B, seq, self.n_kv_heads, self.head_dim)
        else:
            xv = v_override.view(B, seq, self.n_kv_heads, self.head_dim)

        xq, xk = self.rope(xq, xk, start_pos, stream_ids)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores + causal_mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)

        out = torch.matmul(scores, xv)
        return self.wo(out.transpose(1, 2).contiguous().view(B, seq, -1))


# SwiGLU FFN
class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# LIGHTHOUSE ATTENTION — Symmetric Pyramid Selection
class LighthouseWrapper(nn.Module):
    """
    Wraps a standard attention layer with Lighthouse pyramid selection.

    Training-only: during training, apply pyramid pool → score → top-K →
    gather → dense SDPA on subsequence → scatter-back.

    At inference (eval mode), bypass Lighthouse and use dense attention directly.

    Based on Nous Research paper (May 2026). Key design choices:
      - Symmetric Q/K/V pooling (both queries AND keys compressed)
      - Parameter-free L2 norm scorer
      - Stock SDPA on gathered subsequence (no custom sparse kernel)
      - Non-differentiable top-K (gradients flow through scatter → SDPA → gather)
    """
    def __init__(self, args: VERAv6Args):
        super().__init__()
        self.L = args.lighthouse_levels      # pyramid levels
        self.p = args.lighthouse_pool_factor  # pooling factor per level
        self.k = args.lighthouse_top_k       # selection budget

    def _pyramid_pool(self, x, L, p):
        """
        Build L-level pyramid via mean-pooling with factor p.
        x: [B, N, D]
        Returns: list of (tensor, level, original_indices) per level.
        Level 0 = original resolution.
        """
        levels = [(x, 0)]
        current = x
        for ell in range(1, L):
            B, N, D = current.shape
            # Pad if not divisible
            pad = (p - N % p) % p
            if pad > 0:
                current = F.pad(current, (0, 0, 0, pad))
            N_padded = current.shape[1]
            pooled = current.reshape(B, N_padded // p, p, D).mean(dim=2)
            levels.append((pooled, ell))
            current = pooled
        return levels

    def _score_entries(self, q_levels, k_levels):
        """
        Parameter-free L2 norm scorer.
        Score each entry by ‖Q‖₂ + ‖K‖₂ (combined query + key importance).
        Coarser levels inherit max of finer scores.
        Returns flat scores tensor.
        """
        # Level 0: direct L2 norms
        q0 = q_levels[0][0]  # [B, N, D]
        k0 = k_levels[0][0]
        # Per-token scores (averaged across hidden dim for stability)
        q_scores_0 = q0.float().norm(dim=-1)  # [B, N]
        k_scores_0 = k0.float().norm(dim=-1)

        all_scores = []
        all_levels = []

        for ell, (q_ell, _) in enumerate(q_levels):
            if ell == 0:
                scores = q_scores_0 + k_scores_0  # [B, N]
            else:
                # Max-pool from level 0 scores
                B, N0 = q_scores_0.shape
                p_ell = self.p ** ell
                N_ell = q_ell.shape[1]
                # Reshape and max-pool
                padded_n = N_ell * p_ell
                if padded_n > N0:
                    sq = F.pad(q_scores_0, (0, padded_n - N0), value=0)
                    sk = F.pad(k_scores_0, (0, padded_n - N0), value=0)
                else:
                    sq = q_scores_0[:, :padded_n]
                    sk = k_scores_0[:, :padded_n]
                sq = sq.reshape(B, N_ell, p_ell).max(dim=-1).values
                sk = sk.reshape(B, N_ell, p_ell).max(dim=-1).values
                scores = sq + sk

            all_scores.append(scores)
            all_levels.append(torch.full_like(scores, ell, dtype=torch.long))

        return all_scores, all_levels

    def _select_top_k(self, all_scores, all_levels, k):
        """
        Select top-K entries across all pyramid levels.
        Always keep ALL entries from coarsest level (cheap, ensures coverage).
        Distribute remaining budget across finer levels.
        """
        B = all_scores[0].shape[0]
        L = len(all_scores)

        # Coarsest level: keep all
        coarsest_scores = all_scores[-1]
        coarsest_n = coarsest_scores.shape[1]

        # Remaining budget for finer levels
        remaining_k = max(k - coarsest_n, 0)

        if remaining_k == 0 or L <= 1:
            # Only coarsest level
            indices = [torch.arange(coarsest_n, device=coarsest_scores.device
                                    ).unsqueeze(0).expand(B, -1)]
            levels = [all_levels[-1]]
            return indices, levels

        # Concat scores from finer levels (0..L-2)
        finer_scores = torch.cat(all_scores[:-1], dim=1)  # [B, total_finer]
        finer_levels = torch.cat(all_levels[:-1], dim=1)

        # Compute cumulative offsets for index tracking
        offsets = []
        cum = 0
        for ell in range(L - 1):
            offsets.append(cum)
            cum += all_scores[ell].shape[1]
        offsets_t = torch.tensor(offsets, device=finer_scores.device)

        # Top-K from finer levels
        actual_k = min(remaining_k, finer_scores.shape[1])
        _, top_indices = finer_scores.topk(actual_k, dim=1)  # [B, actual_k]

        return top_indices, finer_levels, all_scores, coarsest_n

    def forward(self, attention_layer, x, stream_ids, start_pos, causal_mask):
        """
        Wrap attention_layer with Lighthouse selection during training.

        attention_layer: DSPAttention module
        x: [B, seq, dim]
        stream_ids: [B, seq]

        Returns: [B, seq, dim] — attention output with Lighthouse speedup.
        """
        # ── Bypass in eval mode (inference uses dense attention) ──
        if not self.training:
            return attention_layer(x, stream_ids, start_pos, causal_mask)

        B, N, D = x.shape

        # ── 1. Project Q, K, V ──
        xq = attention_layer.wq(x)
        xk = attention_layer.wk(x)
        xv = attention_layer.wv(x)

        # ── 2. Pyramid pool (symmetric Q, K, V) ──
        q_levels = self._pyramid_pool(xq, self.L, self.p)
        k_levels = self._pyramid_pool(xk, self.L, self.p)
        v_levels = self._pyramid_pool(xv, self.L, self.p)

        # ── 3. Score and select ──
        all_scores, all_level_ids = self._score_entries(q_levels, k_levels)

        # For simplicity in this implementation: keep coarsest level fully,
        # top-K from level 0 (finest). Skip intermediate levels if L=3
        # to keep gather logic clean.

        coarsest_q = q_levels[-1][0]   # [B, N/p^(L-1), D]
        coarsest_k = k_levels[-1][0]
        coarsest_v = v_levels[-1][0]
        coarsest_n = coarsest_q.shape[1]

        # Top-K from finest level (level 0)
        finest_scores = all_scores[0]  # [B, N]
        actual_k = min(self.k, N)
        _, top_idx = finest_scores.topk(actual_k, dim=1)  # [B, k]
        top_idx_sorted, _ = top_idx.sort(dim=1)  # causal ordering

        # ── 4. Gather selected entries ──
        idx_expand = top_idx_sorted.unsqueeze(-1).expand(-1, -1, xq.shape[-1])
        sel_q = torch.gather(xq, 1, idx_expand)  # [B, k, D_q]
        sel_k = torch.gather(xk, 1, idx_expand[:, :, :xk.shape[-1]])
        sel_v = torch.gather(xv, 1, idx_expand[:, :, :xv.shape[-1]])

        # Concat with coarsest level: [B, k + coarsest_n, D]
        gathered_q = torch.cat([sel_q, coarsest_q], dim=1)
        gathered_k = torch.cat([sel_k, coarsest_k], dim=1)
        gathered_v = torch.cat([sel_v, coarsest_v], dim=1)

        # Also gather stream_ids for RoPE
        sel_sids = torch.gather(stream_ids, 1, top_idx_sorted)
        coarsest_sids = torch.zeros(B, coarsest_n,
                                     device=stream_ids.device,
                                     dtype=stream_ids.dtype)
        gathered_sids = torch.cat([sel_sids, coarsest_sids], dim=1)

        S = gathered_q.shape[1]

        # ── 5. Reshape for attention (reuse DSP RoPE) ──
        n_h = attention_layer.n_heads
        n_kv = attention_layer.n_kv_heads
        hd = attention_layer.head_dim
        n_rep = attention_layer.n_rep

        gq = gathered_q.view(B, S, n_h, hd)
        gk = gathered_k.view(B, S, n_kv, hd)
        gv = gathered_v.view(B, S, n_kv, hd)

        # Apply RoPE
        gq, gk = attention_layer.rope(gq, gk, start_pos, gathered_sids)

        gk = repeat_kv(gk, n_rep)
        gv = repeat_kv(gv, n_rep)

        gq = gq.transpose(1, 2)
        gk = gk.transpose(1, 2)
        gv = gv.transpose(1, 2)

        # Build causal mask for gathered sequence
        # Simplified: use standard causal mask on the sorted subsequence
        g_mask = torch.full((S, S), float("-inf"), device=x.device, dtype=torch.float32)
        g_mask = torch.triu(g_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(gq, gk.transpose(2, 3)) / math.sqrt(hd)
        scores = scores + g_mask
        attn = F.softmax(scores.float(), dim=-1).type_as(gq)
        attn = attention_layer.dropout(attn)
        out = torch.matmul(attn, gv)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = attention_layer.wo(out)

        # ── 6. Scatter-back to full sequence ──
        output = torch.zeros(B, N, out.shape[-1], device=x.device, dtype=out.dtype)

        # Fine-level entries: direct scatter
        fine_out = out[:, :actual_k]
        idx_out = top_idx_sorted.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        output.scatter_add_(1, idx_out, fine_out)

        # Coarsest level: distribute to p^(L-1) positions each (with shift)
        coarse_out = out[:, actual_k:]  # [B, coarsest_n, D]
        p_L = self.p ** (self.L - 1)
        for i in range(coarsest_n):
            # Scatter to positions [i*p_L .. (i+1)*p_L - 1], capped at N
            start_idx = i * p_L
            end_idx = min(start_idx + p_L, N)
            if start_idx >= N:
                break
            output[:, start_idx:end_idx] += coarse_out[:, i:i+1].expand(
                -1, end_idx - start_idx, -1)

        # Count contributions for averaging (avoid double-counting)
        counts = torch.zeros(B, N, 1, device=x.device, dtype=output.dtype)
        counts.scatter_add_(1, idx_out[:, :, :1],
                           torch.ones_like(idx_out[:, :, :1], dtype=output.dtype))
        for i in range(coarsest_n):
            start_idx = i * p_L
            end_idx = min(start_idx + p_L, N)
            if start_idx >= N:
                break
            counts[:, start_idx:end_idx] += 1.0
        counts = counts.clamp(min=1.0)
        output = output / counts

        return output


# Perception Layer (v5 + Lighthouse)

class PerceptionLayer(nn.Module):
    def __init__(self, args: VERAv6Args, use_lighthouse: bool = False):
        super().__init__()
        self.attention = DSPAttention(args)
        self.ffn = SwiGLUFFN(args.dim, args.ffn_hidden)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

        self.use_lighthouse = use_lighthouse
        if use_lighthouse:
            self.lighthouse = LighthouseWrapper(args)
        else:
            self.lighthouse = None

    def forward(self, x, stream_ids, start_pos=0, causal_mask=None):
        if self.use_lighthouse and self.lighthouse is not None:
            # Lighthouse wraps attention (handles train/eval mode internally)
            attn_out = self.lighthouse(
                self.attention, self.attn_norm(x), stream_ids,
                start_pos, causal_mask,
            )
        else:
            attn_out = self.attention(
                self.attn_norm(x), stream_ids, start_pos, causal_mask,
            )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ★ MELT Gated KV Cache for Reasoning
class MELTGatedKVState(nn.Module):
    """
    Implements MELT's gated latent state for KV cache.

    Instead of appending KV entries each reasoning iteration (O(seq × iters)),
    maintains a latent state h ∈ R^dim that is updated via GRU-like gating:
      z_t   = σ(x_t W_z + h_{t-1} U_z + b_z)
      h_t   = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ tanh(W_h x_t)
      K_t, V_t = h_t W_K, h_t W_V        ← actually consumed by attention

    Key fix (v6.1): state is in full hidden dimension (dim, not kv_dim),
    and W_k / W_v project state → kv_dim. The output (k_proj, v_proj) is
    returned to be consumed by DSPAttention via k_override / v_override.
    Without this, the previous version computed h_t but never injected it,
    so the parameters were dead.
    """
    def __init__(self, dim: int, kv_dim: int):
        """
        dim: model hidden dimension (state lives here)
        kv_dim: n_kv_heads × head_dim (K/V projection output)
        """
        super().__init__()
        self.dim = dim
        self.kv_dim = kv_dim

        # Gate parameters (full-dim ↔ full-dim, GRU-like)
        self.W_z = nn.Linear(dim, dim, bias=False)
        self.U_z = nn.Linear(dim, dim, bias=False)
        self.b_z = nn.Parameter(torch.zeros(dim))

        # Candidate hidden state projection
        self.W_h = nn.Linear(dim, dim, bias=False)

        # Output projections: state → K, state → V
        self.W_k = nn.Linear(dim, kv_dim, bias=False)
        self.W_v = nn.Linear(dim, kv_dim, bias=False)

        nn.init.zeros_(self.b_z)
        nn.init.normal_(self.W_z.weight, std=0.02)
        nn.init.normal_(self.U_z.weight, std=0.02)
        nn.init.normal_(self.W_h.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)

    def init_state(self, batch_size, seq_len, device, dtype):
        """Initialize latent state to zeros (first iteration has no history)."""
        return torch.zeros(batch_size, seq_len, self.dim,
                           device=device, dtype=dtype)

    def update(self, x, h_prev):
        """
        x:      [B, seq, dim]  current input to reasoning block
        h_prev: [B, seq, dim]  previous latent state

        Returns:
          h_new: [B, seq, dim]   updated latent state
          k:     [B, seq, kv_dim] K projection of h_new
          v:     [B, seq, kv_dim] V projection of h_new
          z:     [B, seq, dim]   gate values (diagnostics)
        """
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev) + self.b_z)
        h_cand = torch.tanh(self.W_h(x))
        h_new = z * h_prev + (1.0 - z) * h_cand
        k = self.W_k(h_new)
        v = self.W_v(h_new)
        return h_new, k, v, z


# TTM — Tree Thought Memory (v5, FIXED router)
class TreeThoughtMemory(nn.Module):
    def __init__(self, args: VERAv6Args):
        super().__init__()
        self.n_branches = args.n_branches
        self.dim = args.dim

        self.branch_proj = nn.ModuleList([
            nn.Sequential(
                RMSNorm(args.dim, args.norm_eps),
                nn.Linear(args.dim, args.dim // 2, bias=False),
                nn.SiLU(),
                nn.Linear(args.dim // 2, args.dim, bias=False),
            ) for _ in range(args.n_branches)
        ])

        # FIX v6: router takes pooled mean (dim) instead of flat concat (dim*n_branches)
        # Iteration-conditioned prior routing (input-agnostic). Problem-aware
        # routing is done position-wise in CausalThoughtBridge (causal).
        self.router = nn.Sequential(
            nn.Linear(args.dim, args.branch_router_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(args.branch_router_hidden, args.n_branches, bias=True),
        )
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, thought_state, hard=False):
        """Returns (merged [B,n_t,D], weights [B,n_br], branches [B,n_t,n_br,D])."""
        branches = torch.stack([
            thought_state + proj(thought_state) for proj in self.branch_proj
        ], dim=2)  # [B, n_thought, n_branches, dim]

        # Prior router: iteration-conditioned mixture (no sequence input)
        pooled = branches.mean(dim=(1, 2))  # [B, dim]
        logits = self.router(pooled)         # [B, n_branches]

        if hard and not self.training:
            idx = logits.argmax(dim=-1)
            B = idx.shape[0]
            merged = branches[torch.arange(B, device=idx.device), :, idx, :]
            weights = F.one_hot(idx, self.n_branches).float()
        else:
            weights = F.softmax(logits, dim=-1)
            w = weights.view(-1, 1, self.n_branches, 1)
            merged = (w * branches).sum(dim=2)

        return merged, weights, branches


# Thought Tokens (v5 base + TTM)
class ThoughtTokens(nn.Module):
    def __init__(self, args: VERAv6Args):
        super().__init__()
        self.n_thought = args.n_thought_tokens
        self.dim = args.dim
        self.use_tree = args.use_tree_memory

        self.thought_embeds = nn.Parameter(
            torch.randn(self.n_thought, args.dim) * 0.02)
        total_iters = args.n_reasoning_blocks * args.max_reasoning_iters
        self.iter_embeds = nn.Parameter(
            torch.randn(total_iters, args.dim) * (0.02 / math.sqrt(2)))
        self.thought_norm = RMSNorm(args.dim)

        if self.use_tree:
            self.tree = TreeThoughtMemory(args)

    def init_state(self, B, device, dtype):
        return self.thought_embeds.unsqueeze(0).expand(B, -1, -1).clone().to(
            device=device, dtype=dtype)

    def step_thought(self, prev, global_iter, hard=False):
        """Returns (thought_state, prior_weights, branches|None)."""
        iter_emb = self.iter_embeds[global_iter].view(1, 1, -1)
        normed = self.thought_norm(prev) + iter_emb
        if self.use_tree:
            return self.tree(normed, hard=hard)
        return normed, None, None

    def inject(self, x, thought_state):
        return torch.cat([thought_state, x], dim=1)

    def strip(self, x):
        return x[:, self.n_thought:], x[:, :self.n_thought]


# IVH — Inline Verifier Head (v5, unchanged)
class VerifierHead(nn.Module):
    def __init__(self, args: VERAv6Args):
        super().__init__()
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.net = nn.Sequential(
            nn.Linear(args.dim, args.verifier_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(args.verifier_hidden, 1, bias=True),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, thought_state):
        pooled = self.norm(thought_state).mean(dim=1)
        return torch.sigmoid(self.net(pooled).squeeze(-1))


# EHG — Energy Halt Gate (v5, FIXED: remove 0.25 cap)

class EnergyHaltGate(nn.Module):
    """
    v6 FIX: Previous v5 used gate = σ(-E/τ) × σ(-|ΔE|/τ), which caps
    the gate at 0.25 (product of two [0,1] values). This severely limits
    reasoning block updates.

    v6: Use single energy-based gate with convergence bonus:
      gate = σ(-(E + α·|ΔE|) / τ)
    This allows full [0,1] range. Low E AND low ΔE → high gate (halt).
    """
    def __init__(self, args: VERAv6Args):
        super().__init__()
        self.tau = args.energy_temperature
        self.convergence_alpha = 0.5  # weight for ΔE term

        self.energy_net = nn.Sequential(
            nn.Linear(args.dim * 2, args.dim // 4, bias=False),
            nn.SiLU(),
            nn.Linear(args.dim // 4, 1, bias=True),
        )
        # ANTI-COLLAPSE: gate = σ(-E/τ). Khởi E âm (bias âm) → gate khởi điểm CAO
        # (≈σ(1.5/τ)≈0.82 với τ=1) để reasoning blocks nhận gradient mạnh từ step 0,
        # tránh sụp về residual-passthrough trước khi gate-reg kịp tác dụng.
        # (Chỉ ảnh hưởng pretrain from-scratch; load ckpt sẽ ghi đè bias.)
        nn.init.constant_(self.energy_net[-1].bias, -1.5 * args.energy_temperature)

    def forward(self, residual, h, prev_energy=None):
        # CAUSALITY FIX: compute energy/gate PER POSITION (no mean over seq).
        # The old code pooled residual.mean(dim=1)/h.mean(dim=1) over the whole
        # sequence → one scalar gate applied to every position, so changing a
        # future token shifted the pooled mean and leaked into the prefix
        # (residual future-token leak ~0.08 logits). Per-position gate[t] depends
        # only on residual[t]/h[t]; since h[t] is causal, gate[t] is causal too.
        E = self.energy_net(torch.cat([residual, h], dim=-1)).squeeze(-1)  # [B, seq]

        if prev_energy is not None:
            delta = (prev_energy - E).abs()
            # single sigmoid with combined signal (full [0,1] range)
            gate = torch.sigmoid(-(E + self.convergence_alpha * delta) / self.tau)
        else:
            gate = torch.sigmoid(-E / self.tau)

        return gate.unsqueeze(-1), E   # gate [B, seq, 1], E [B, seq]


# Reasoning Block (v5 + MELT gated KV)
class ReasoningBlock(nn.Module):
    """
    Reasoning block with:
    - DSP Attention (dual-stream RoPE)
    - MELT Gated KV (constant memory across iterations)
    - Energy Halt Gate
    """
    def __init__(self, args: VERAv6Args):
        super().__init__()
        self.attention = DSPAttention(args)
        self.ffn = SwiGLUFFN(args.dim, args.ffn_hidden)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

        self.halt_mode = args.halt_mode
        if args.halt_mode == "energy":
            self.halt = EnergyHaltGate(args)
        else:
            self.halt = nn.Sequential(
                nn.Linear(args.dim * 2, args.dim // 4, bias=False),
                nn.SiLU(),
                nn.Linear(args.dim // 4, 1, bias=True),
            )
            nn.init.zeros_(self.halt[-1].bias)

        # ★ MELT Gated KV State
        self.use_melt = args.use_melt_kv
        if self.use_melt:
            kv_dim = args.n_kv_heads * (args.dim // args.n_heads)
            self.melt_kv = MELTGatedKVState(args.dim, kv_dim)

    def forward(self, x, stream_ids, causal_mask=None, prev_energy=None,
                melt_state=None):
        """
        Returns:
          x_out: updated hidden state
          gate: halt gate value
          E: energy (or None if sigmoid mode)
          melt_state_new: updated MELT latent state (or None)
          melt_gate: MELT gate values for diagnostics (or None)
        """
        residual = x
        x_norm = self.attn_norm(residual)

        # ★ MELT: update gated latent state BEFORE attention, derive K, V
        melt_state_new = None
        melt_gate = None
        k_inj = None
        v_inj = None
        if self.use_melt and melt_state is not None:
            melt_state_new, k_inj, v_inj, melt_gate = self.melt_kv.update(
                x_norm, melt_state,
            )

        # Attention — when MELT is active, K/V come from gated state h_t.
        # Q is still computed from x_norm (queries are "where to look").
        attn_out = self.attention(
            x_norm, stream_ids, 0, causal_mask,
            k_override=k_inj, v_override=v_inj,
        )
        h = residual + attn_out
        h = h + self.ffn(self.ffn_norm(h))

        # Halt gate
        if self.halt_mode == "energy":
            gate, E = self.halt(residual, h, prev_energy)
            x_out = gate * h + (1.0 - gate) * residual
            return x_out, gate, E, melt_state_new, melt_gate
        else:
            gate_input = torch.cat([residual, h], dim=-1)
            gate = torch.sigmoid(self.halt(gate_input))
            x_out = gate * h + (1.0 - gate) * residual
            return x_out, gate, None, melt_state_new, melt_gate


# ═══════════════════════════════════════════════════════════════════
# Causal Thought Bridge — input-dependent thoughts without future leak
# ═══════════════════════════════════════════════════════════════════

class CausalThoughtBridge(nn.Module):
    """3-phase thought↔real coupling: causal + input-dependent + problem-aware tree.

    Phase 1 Perception: real-only causal (outside this module).

    Phase 2 Thought reads real (causal):
      s_i = mean(h[0:i+1])

    Phase 3 Real reads thought (causal, single channel — joint R→T is OFF):
      If TTM branches provided [B,n_t,n_br,D]:
        # problem-aware soft routing per position (uses only s_i → causal)
        w_i = softmax(Router([pool(thought_prior); s_i]))
        t_i = Σ_b w_i[b] · mean_n(branches[:,b,:])
      else:
        t_i = mean(thought_state)
      h_i ← h_i + W_o(W_t t_i + W_s s_i)   # W_o zero-init → identity start

    Shared TTM prior router stays iteration-conditioned (no full-seq inject),
    so early positions never read future via a contaminated thought bank.

    Verifier (sequence-level, auxiliary): thought + W_v s_{S-1}
      — full-seq pool is OK here; does not write back into early logits.
      — at autoregressive decode, s_{S-1} is the generated prefix so far.
    """

    def __init__(self, args: VERAv6Args):
        super().__init__()
        d = args.dim
        self.n_branches = args.n_branches if args.use_tree_memory else 1
        self.prefix_proj = nn.Linear(d, d, bias=False)
        self.thought_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.verifier_proj = nn.Linear(d, d, bias=False)
        # Problem-aware position-wise branch router: [thought_pool || s_i] → n_br
        self.pos_router = nn.Sequential(
            nn.Linear(d * 2, args.branch_router_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(args.branch_router_hidden, self.n_branches, bias=True),
        )
        nn.init.zeros_(self.pos_router[-1].bias)

        # Stable continue-pretrain: residual path starts as identity
        nn.init.zeros_(self.out_proj.weight)
        nn.init.normal_(self.prefix_proj.weight, std=0.02)
        nn.init.normal_(self.thought_proj.weight, std=0.02)
        nn.init.normal_(self.verifier_proj.weight, std=0.02)

    @staticmethod
    def causal_prefix_mean(x: torch.Tensor) -> torch.Tensor:
        """x [B,S,D] → prefix mean s_i = mean(x[0:i+1]) [B,S,D]."""
        csum = torch.cumsum(x, dim=1)
        n = torch.arange(
            1, x.shape[1] + 1, device=x.device, dtype=x.dtype
        ).view(1, -1, 1)
        return csum / n

    def apply_to_real(
        self,
        thought_state: torch.Tensor,
        x: torch.Tensor,
        branches: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Phase 2+3: causal residual; optional problem-aware tree routing."""
        prefix = self.causal_prefix_mean(x)           # [B,S,D]
        B, S, D = x.shape

        if branches is not None and branches.size(2) == self.n_branches:
            # branches: [B, n_t, n_br, D] → mean over thought slots
            bmean = branches.mean(dim=1)              # [B, n_br, D]
            t_pool = thought_state.mean(dim=1)        # [B, D] prior summary
            t_exp = t_pool.unsqueeze(1).expand(B, S, D)
            rout_in = torch.cat([t_exp, prefix], dim=-1)   # [B,S,2D]
            w = F.softmax(self.pos_router(rout_in), dim=-1)  # [B,S,n_br]
            t_pos = torch.einsum("bsn,bnd->bsd", w, bmean)   # [B,S,D]
        else:
            t_pos = thought_state.mean(dim=1, keepdim=True).expand(B, S, D)

        ctx = self.thought_proj(t_pos) + self.prefix_proj(prefix)
        return x + self.out_proj(ctx)

    def thought_for_verifier(
        self, thought_state: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Sequence-level input-dependent thought for the verifier head.

        Uses s_{S-1} (full teacher-forced seq at train; generated prefix at decode).
        Does NOT write into the per-position LM residual path.
        """
        prefix = self.causal_prefix_mean(x)
        s_last = prefix[:, -1:, :]                             # [B,1,D]
        return thought_state + self.verifier_proj(s_last)


# Mask Cache
class _MaskCache:
    def __init__(self):
        self._c: Dict[tuple, torch.Tensor] = {}
        self._t: Dict[tuple, torch.Tensor] = {}

    def causal(self, n, device):
        k = (n, str(device))
        if k not in self._c:
            m = torch.full((n, n), float("-inf"), device=device)
            self._c[k] = torch.triu(m, diagonal=1).unsqueeze(0).unsqueeze(0)
        return self._c[k]

    def thought_causal(self, n_t, n_r, device):
        """Joint self-attn mask on [thought | real].

        3-phase design (v6.2):
          Phase 1 Perception: real-only causal (separate mask).
          Phase 2 Thought→Real: NOT via this joint matrix (would need
              per-position thought banks). Instead CausalThoughtBridge
              does causal prefix read + position-wise write.
          Phase 3 Real→Thought + Real→Real: allowed here.

        Joint matrix rules (v6.2.1 single-channel thought→real via bridge):
          T→T  full     (shared latent prior / TTM)
          T→R  blocked  (no full-seq absorb / future leak)
          R→T  blocked  (avoid dual path; thought→real only via bridge residual)
          R→R  causal
        """
        k = (n_t, n_r, str(device), "v621")
        if k not in self._t:
            total = n_t + n_r
            m = torch.full((total, total), float("-inf"), device=device)
            m[:n_t, :n_t] = 0.0          # thought ↔ thought
            # T→R blocked, R→T blocked — only bridge couples thought↔real
            for i in range(n_r):
                m[n_t + i, n_t:n_t + i + 1] = 0.0   # real → real (causal)
            self._t[k] = m.unsqueeze(0).unsqueeze(0)
        return self._t[k]


# ═══════════════════════════════════════════════════════════════════
# Stream tagging utility
# ═══════════════════════════════════════════════════════════════════

def build_operator_token_set(tokenizer, extra_chars=""):
    op_chars = set("0123456789+-*/=<>(){}[]^√∑∫πθ°%≈≠≤≥" + extra_chars)
    op_ids = set()
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for tok_str, tok_id in vocab.items():
        cleaned = tok_str.lstrip("▁ ").strip()
        if cleaned and all(c in op_chars for c in cleaned):
            op_ids.add(tok_id)
    return op_ids


def tag_streams(tokens, operator_ids):
    if not operator_ids:
        return torch.zeros_like(tokens)
    op_tensor = torch.tensor(sorted(operator_ids),
                             device=tokens.device, dtype=tokens.dtype)
    return (tokens.unsqueeze(-1) == op_tensor).any(dim=-1).long()



#Main Model
class VERAPsi(nn.Module):
    def __init__(self, args: VERAv6Args):
        super().__init__()
        assert args.vocab_size > 0
        self.args = args

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # Perception layers: some with Lighthouse, some dense
        self.perception_layers = nn.ModuleList()
        for i in range(args.n_perception_layers):
            use_lh = (args.use_lighthouse
                      and i >= args.lighthouse_skip_first
                      and i < args.n_perception_layers - args.lighthouse_skip_last)
            self.perception_layers.append(PerceptionLayer(args, use_lighthouse=use_lh))

        self.mid_norm = RMSNorm(args.dim, args.norm_eps)
        self.thought_tokens = ThoughtTokens(args)

        # v6.2: causal input-dependent thought↔real bridge
        self.use_causal_thought_bridge = bool(
            getattr(args, "use_causal_thought_bridge", True)
            and getattr(args, "thought_read_mode", "causal_cross") != "independent"
        )
        self.thought_bridge = (
            CausalThoughtBridge(args) if self.use_causal_thought_bridge else None
        )

        self.reasoning_blocks = nn.ModuleList(
            [ReasoningBlock(args) for _ in range(args.n_reasoning_blocks)]
        )

        if args.use_verifier:
            self.verifier = VerifierHead(args)
        else:
            self.verifier = None

        self.final_norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._mask_cache = _MaskCache()
        self._init_weights()
        # Re-apply residual zero-init after global _init_weights (would re-randn)
        if self.thought_bridge is not None:
            nn.init.zeros_(self.thought_bridge.out_proj.weight)
            nn.init.zeros_(self.thought_bridge.pos_router[-1].bias)

    def _apply_thought_bridge_real(self, thought_state, x, branches=None):
        """Phase 2+3: causal input-dependent residual into real states."""
        if self.thought_bridge is None:
            return x
        return self.thought_bridge.apply_to_real(thought_state, x, branches)

    def _thought_for_verifier(self, thought_state, x):
        if self.thought_bridge is None or self.verifier is None:
            return thought_state
        return self.thought_bridge.thought_for_verifier(thought_state, x)

    def _init_weights(self):
        n_layers = self.args.n_perception_layers + self.args.n_reasoning_blocks
        scale = 1.0 / math.sqrt(2.0 * n_layers) if self.args.init_scale_with_depth else 1.0

        for name, p in self.named_parameters():
            if any(s in name for s in [
                "thought_embeds", "iter_embeds", "halt", "energy_net",
                "router", "branch_proj", "verifier", "mix_logits",
                "norm.weight",
                "melt_kv.W_z", "melt_kv.U_z", "melt_kv.W_h",
                "melt_kv.W_k", "melt_kv.W_v", "melt_kv.b_z",
            ]):
                continue
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
                if "wo.weight" in name or "ffn.down.weight" in name:
                    p.data.mul_(scale)
            elif p.dim() == 1 and "norm" not in name:
                nn.init.zeros_(p)

    def _maybe_ckpt(self, fn, *a, **kw):
        if self.training and self.args.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                fn, *a, use_reentrant=False, **kw)
        return fn(*a, **kw)

    def _build_thought_stream(self, stream_ids):
        B = stream_ids.shape[0]
        return torch.zeros(B, self.args.n_thought_tokens,
                          device=stream_ids.device, dtype=stream_ids.dtype)

    def forward(self, tokens, stream_ids, start_pos=0):
        B, seq = tokens.shape
        x = self.tok_embeddings(tokens)

        # Phase 1: Perception (with optional Lighthouse)
        causal = self._mask_cache.causal(seq, x.device)
        for layer in self.perception_layers:
            x = self._maybe_ckpt(layer, x, stream_ids, start_pos, causal)
        x = self.mid_norm(x)

        # Phase 2: Reasoning (with MELT + TTM + EHG)
        n_t = self.args.n_thought_tokens
        thought_stream = self._build_thought_stream(stream_ids)
        full_stream = torch.cat([thought_stream, stream_ids], dim=1)
        thought_mask = self._mask_cache.thought_causal(n_t, seq, x.device)

        thought_state = self.thought_tokens.init_state(B, x.device, x.dtype)
        prev_energy = None

        # ★ Init MELT states per reasoning block
        melt_states = {}
        if self.args.use_melt_kv:
            for blk_idx, blk in enumerate(self.reasoning_blocks):
                if blk.use_melt:
                    kv_dim = blk.melt_kv.kv_dim
                    total_seq = n_t + seq
                    melt_states[blk_idx] = blk.melt_kv.init_state(
                        B, total_seq, x.device, x.dtype)

        for blk_idx, block in enumerate(self.reasoning_blocks):
            for it in range(self.args.max_reasoning_iters):
                g_iter = blk_idx * self.args.max_reasoning_iters + it

                # TTM step (shared latent prior; branches for problem-aware routing)
                thought_state, _, branches = self.thought_tokens.step_thought(
                    thought_state, g_iter, hard=(not self.training))

                # Phase 2+3: causal prefix + position-wise tree routing → real
                x = self._apply_thought_bridge_real(thought_state, x, branches)

                x_with = self.thought_tokens.inject(x, thought_state)

                # Joint mask: T↔T only among thoughts; R→R causal; no T↔R
                ms = melt_states.get(blk_idx)
                x_with, _, E, ms_new, _ = self._maybe_ckpt(
                    block, x_with, full_stream, thought_mask,
                    prev_energy, ms)

                x, thought_state = self.thought_tokens.strip(x_with)
                if E is not None:
                    prev_energy = E
                if ms_new is not None:
                    melt_states[blk_idx] = ms_new

        x = self.final_norm(x)
        return self.output(x).float()

    def forward_with_loss(self, tokens, stream_ids, labels=None,
                          correctness=None, start_pos=0):
        B, seq = tokens.shape
        x = self.tok_embeddings(tokens)

        causal = self._mask_cache.causal(seq, x.device)
        for layer in self.perception_layers:
            x = layer(x, stream_ids, start_pos, causal)
        x = self.mid_norm(x)

        n_t = self.args.n_thought_tokens
        thought_stream = self._build_thought_stream(stream_ids)
        full_stream = torch.cat([thought_stream, stream_ids], dim=1)
        thought_mask = self._mask_cache.thought_causal(n_t, seq, x.device)

        thought_state = self.thought_tokens.init_state(B, x.device, x.dtype)
        prev_energy = None
        energies = []
        verifier_scores = []
        gate_values = []
        melt_gate_values = []

        melt_states = {}
        if self.args.use_melt_kv:
            for bi, blk in enumerate(self.reasoning_blocks):
                if blk.use_melt:
                    total_seq = n_t + seq
                    melt_states[bi] = blk.melt_kv.init_state(
                        B, total_seq, x.device, x.dtype)

        for bi, block in enumerate(self.reasoning_blocks):
            for it in range(self.args.max_reasoning_iters):
                g_iter = bi * self.args.max_reasoning_iters + it
                thought_state, _, branches = self.thought_tokens.step_thought(
                    thought_state, g_iter, hard=False)
                x = self._apply_thought_bridge_real(thought_state, x, branches)
                x_with = self.thought_tokens.inject(x, thought_state)

                ms = melt_states.get(bi)
                x_with, gate, E, ms_new, mg = block(
                    x_with, full_stream, thought_mask, prev_energy, ms)
                x, thought_state = self.thought_tokens.strip(x_with)

                gate_values.append(gate.detach())
                if E is not None:
                    energies.append(E)
                    prev_energy = E
                if ms_new is not None:
                    melt_states[bi] = ms_new
                if mg is not None:
                    melt_gate_values.append(mg.detach())
                if self.verifier is not None:
                    v_in = self._thought_for_verifier(thought_state, x)
                    verifier_scores.append(self.verifier(v_in))

        x = self.final_norm(x)
        logits = self.output(x).float()

        # ── Losses ──
        if labels is None:
            lm_logits = logits[:, :-1].reshape(-1, self.args.vocab_size)
            lm_targets = tokens[:, 1:].reshape(-1)
        else:
            lm_logits = logits.reshape(-1, self.args.vocab_size)
            lm_targets = labels.reshape(-1)
        lm_loss = F.cross_entropy(lm_logits, lm_targets, ignore_index=-100)

        info = {"lm_loss": lm_loss.item()}
        total = lm_loss

        # Verifier deep supervision
        if self.verifier is not None and correctness is not None:
            v_loss = sum(
                F.binary_cross_entropy(s, correctness.float())
                for s in verifier_scores
            ) / max(len(verifier_scores), 1)
            total = total + self.args.verifier_loss_weight * v_loss
            info["verifier_loss"] = v_loss.item()
            info["verifier_loss_tensor"] = v_loss          # FIX (Bug E)
            info["verifier_final_score"] = verifier_scores[-1].mean().item()

        # Energy regularization
        if energies and self.args.energy_reg_weight > 0:
            e_stack = torch.stack(energies, dim=0)
            deltas = e_stack[1:] - e_stack[:-1]
            e_reg = F.relu(deltas).mean()
            total = total + self.args.energy_reg_weight * e_reg
            info["energy_reg"] = e_reg.item()
            info["energy_reg_tensor"] = e_reg              # FIX (Bug E)
            info["energy_trace"] = [e.mean().item() for e in energies]

        # ── Expose reasoning-progression signals for RL intrinsic reward ──
        # (cheap, detached; populated regardless of `correctness`/loss flags so
        #  the RL stage can read the model's built-in reasoning trace.)
        if verifier_scores:
            # per-iter verifier confidence, mean over batch → [n_iters]
            info["verifier_scores_tensor"] = torch.stack(
                [s.detach().float().mean() for s in verifier_scores])
        if energies:
            # per-iter energy, mean over batch → [n_iters]
            info["energy_trace_tensor"] = torch.stack(
                [e.detach().float().mean() for e in energies])

        info["lm_loss_tensor"] = lm_loss                    # FIX (Bug E)

        # Gate diagnostics
        if gate_values:
            all_g = torch.stack(gate_values, dim=0)
            info["gate_mean"] = all_g.mean().item()
            info["gate_std"] = all_g.std().item()

        # MELT diagnostics
        if melt_gate_values:
            all_mg = torch.cat([g.reshape(-1) for g in melt_gate_values])
            info["melt_gate_mean"] = all_mg.mean().item()
            info["melt_gate_std"] = all_mg.std().item()

        return logits, total, info

    # ── Adaptive inference (early-exit) ──
    @torch.no_grad()
    def adaptive_forward(self, tokens, stream_ids,
                         halt_threshold=0.7, max_iters=None):
        was_training = self.training
        self.eval()
        try:
            B, seq = tokens.shape
            x = self.tok_embeddings(tokens)
            causal = self._mask_cache.causal(seq, x.device)
            for layer in self.perception_layers:
                x = layer(x, stream_ids, 0, causal)
            x = self.mid_norm(x)

            n_t = self.args.n_thought_tokens
            ts = self._build_thought_stream(stream_ids)
            fs = torch.cat([ts, stream_ids], dim=1)
            tm = self._mask_cache.thought_causal(n_t, seq, x.device)

            thought_state = self.thought_tokens.init_state(B, x.device, x.dtype)
            prev_energy = None
            iters_used = 0
            max_total = max_iters or (
                self.args.n_reasoning_blocks * self.args.max_reasoning_iters)

            melt_states = {}
            if self.args.use_melt_kv:
                for bi, blk in enumerate(self.reasoning_blocks):
                    if blk.use_melt:
                        total_seq = n_t + seq
                        melt_states[bi] = blk.melt_kv.init_state(
                            B, total_seq, x.device, x.dtype)

            for bi, block in enumerate(self.reasoning_blocks):
                for it in range(self.args.max_reasoning_iters):
                    if iters_used >= max_total:
                        break
                    g_iter = bi * self.args.max_reasoning_iters + it
                    thought_state, _, branches = self.thought_tokens.step_thought(
                        thought_state, g_iter, hard=True)
                    x = self._apply_thought_bridge_real(thought_state, x, branches)
                    x_with = self.thought_tokens.inject(x, thought_state)
                    ms = melt_states.get(bi)
                    x_with, _, E, ms_new, _ = block(
                        x_with, fs, tm, prev_energy, ms)
                    x, thought_state = self.thought_tokens.strip(x_with)
                    iters_used += 1
                    if E is not None:
                        prev_energy = E
                    if ms_new is not None:
                        melt_states[bi] = ms_new

                    if self.verifier is not None:
                        v_in = self._thought_for_verifier(thought_state, x)
                        score = self.verifier(v_in).mean().item()
                        if score > halt_threshold:
                            break

            x = self.final_norm(x)
            return self.output(x).float(), iters_used
        finally:
            if was_training:
                self.train()

    # ── Utilities ──
    def num_params(self, exclude_embedding=False):
        seen = set()
        n = 0
        for p in self.parameters():
            ptr = p.data.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            n += p.numel()
        if exclude_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n

    def print_architecture(self):
        a = self.args
        n_lh = sum(1 for l in self.perception_layers
                   if l.use_lighthouse)

        print("\n" + "=" * 70)
        print("  Nona v6 — MELT + Lighthouse + Verifier + Energy Gate")
        print("=" * 70)
        print(f"  Total params:    {self.num_params() / 1e6:.1f}M")
        print(f"  Non-embed:       {self.num_params(True) / 1e6:.1f}M")
        print(f"  dim={a.dim}, heads={a.n_heads}, kv={a.n_kv_heads}")

        print(f"\n  ① DSP — Dual-Stream RoPE")
        print(f"     Lang θ: {a.rope_thetas_lang}")
        print(f"     Op   θ: {a.rope_thetas_op}")

        print(f"\n  ② Lighthouse Attention (training-only)")
        print(f"     Applied to {n_lh}/{a.n_perception_layers} perception layers")
        print(f"     L={a.lighthouse_levels}, p={a.lighthouse_pool_factor}, "
              f"k={a.lighthouse_top_k}")

        print(f"\n  ③ TTM — Tree Thought Memory")
        print(f"     {a.n_thought_tokens} tokens, {a.n_branches}-way branching")

        print(f"\n  ④ MELT Gated KV Cache")
        print(f"     Gated latent state in reasoning blocks")
        print(f"     Memory: O(seq) constant vs O(seq × iters) without MELT")

        if a.use_verifier:
            print(f"\n  ⑤ IVH — Inline Verifier (deep supervision)")

        print(f"\n  ⑥ EHG — Energy Halt Gate (FIXED: full [0,1] range)")

        print(f"\n  Recursion: {a.n_reasoning_blocks} × {a.max_reasoning_iters} "
              f"= {a.n_reasoning_blocks * a.max_reasoning_iters} max iters")
        print("=" * 70 + "\n")



# Backward-compat alias (pretrain_villama / sft_train / RL_train import
# `VERAArgs`; v6 renamed to VERAv6Args).

VERAArgs = VERAv6Args



# Sanity check


if __name__ == "__main__":
    args = VERAv6Args(
        dim=256, n_heads=8, n_kv_heads=2, vocab_size=1000,
        n_perception_layers=4, n_reasoning_blocks=2,
        max_reasoning_iters=2, n_thought_tokens=8,
        ffn_hidden=512, max_seq_len=128,
        lighthouse_levels=2, lighthouse_pool_factor=2,
        lighthouse_top_k=32,
        lighthouse_skip_first=1, lighthouse_skip_last=1,
    )
    model = VERAPsi(args)
    model.print_architecture()

    B, T = 2, 32
    tokens = torch.randint(0, args.vocab_size, (B, T))
    stream_ids = torch.randint(0, 2, (B, T))
    correctness = torch.tensor([1.0, 0.0])

    print("→ Forward (eval)")
    model.eval()
    logits = model(tokens, stream_ids)
    print(f"  logits: {tuple(logits.shape)}")

    print("\n→ Forward with loss (train)")
    model.train()
    logits, loss, info = model.forward_with_loss(
        tokens, stream_ids, correctness=correctness)
    print(f"  loss: {loss.item():.4f}")
    for k, v in info.items():
        if isinstance(v, list):
            print(f"  {k}: {[round(x, 4) for x in v]}")
        else:
            print(f"  {k}: {v}")

    print("\n→ Backward")
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  {n_grad} params have gradients")

    print("\n→ Adaptive inference")
    model.eval()
    logits, n_used = model.adaptive_forward(tokens, stream_ids)
    print(f"  iterations used: {n_used}")

    print("\n✓ All paths working")