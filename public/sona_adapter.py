"""sona_adapter.py — run the recurrent model (VERAPsi) under DCP execution schedules.

The schedule runner mirrors VERAPsi.forward exactly (eval mode: hard thought routing, causal thought
bridge, MELT KV state carried across applications), but executes an explicit list of (block, iter)
applications. Each application keeps its ORIGINAL global iteration index g = b * n_i + t; repeated
applications reuse the index of the repeated application (Table 1 of the paper).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

TMLR = Path(__file__).resolve().parent  # vera_psi.py is shipped next to this file
sys.path.insert(0, str(TMLR))
import vera_psi as V  # noqa: E402


class SonaTok:
    """Minimal tokenizer wrapper with the interface dcp_public.tokenize expects."""

    class _Enc:
        def __init__(self, ids):
            self.input_ids = ids

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(path))
        self.bos_token_id = self.t.token_to_id("<bos>")

    def __call__(self, text, add_special_tokens=False):
        return self._Enc(self.t.encode(text, add_special_tokens=add_special_tokens).ids)

    def get_vocab(self):
        return self.t.get_vocab()


def load(ckpt_path, device, dtype=torch.bfloat16):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = V.VERAv6Args(**ck["config"])
    a.gradient_checkpointing = False
    a.dropout = 0.0
    m = V.VERAPsi(a)
    res = m.load_state_dict(ck["model"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    m = m.to(device=device, dtype=dtype).eval()
    tok = SonaTok(Path(ckpt_path).parent / "tokenizer.json")
    return tok, m, int(ck.get("step", -1))


class SonaAdapter:
    kind = "sona"

    def __init__(self, model, tok):
        self.m = model
        self.a = model.args
        self.head = model.output
        self.ops = torch.tensor(sorted(V.build_operator_token_set(tok)), dtype=torch.long)
        self.cap = {}
        self.head.register_forward_pre_hook(lambda mod, args: self.cap.__setitem__("z", args[0]))
        self.m.final_norm.register_forward_pre_hook(lambda mod, args: self.cap.__setitem__("h", args[0]))
        self.dyn = None
        self.nb, self.ni = self.a.n_reasoning_blocks, self.a.max_reasoning_iters
        self.N = self.nb * self.ni

    def stream_ids(self, ids):
        return (ids.unsqueeze(-1) == self.ops.to(ids.device)).any(-1).long()

    def configs(self, fracs=None, steps=None):
        pairs = [(b, t) for b in range(self.nb) for t in range(self.ni)]
        cf = [("full", {"plan": pairs})]
        for k in (1, 2, 4):
            cf.append((f"prefix{k}", {"plan": pairs[:k]}))
            cf.append((f"suffix{k}", {"plan": pairs[-k:]}))
            cf.append((f"repeat{k}", {"plan": pairs[:k] + [pairs[k - 1]] * (self.N - k)}))
        # randomized-iteration control: same blocks and application count, global indices
        # permuted within each block (fixed seed)
        rnd = random.Random(0)
        perm = []
        for b in range(self.nb):
            ts = list(range(self.ni))
            while ts == list(range(self.ni)):
                rnd.shuffle(ts)
            perm += [(b, t) for t in ts]
        cf.append(("permute8", {"plan": perm}))
        # extrapolation beyond the trained 8 applications
        cf.append(("extend12", {"plan": pairs + [pairs[-1]] * 4}))
        cf.append(("extend16", {"plan": pairs + [pairs[-1]] * 8}))
        cf.append(("cycle16", {"plan": pairs + pairs[self.ni:]}))
        return cf

    @torch.no_grad()
    def _run(self, ids, cfg, trace=False):
        """Execute schedule cfg["plan"]; returns final real-token state x and, if trace, per-application
        (state after the application, mean gate over real tokens per position)."""
        m, a = self.m, self.a
        self.cap.clear()
        tokens = ids
        sids = self.stream_ids(tokens)
        B, seq = tokens.shape
        x = m.tok_embeddings(tokens)
        causal = m._mask_cache.causal(seq, x.device)
        for layer in m.perception_layers:
            x = layer(x, sids, 0, causal)
        x = m.mid_norm(x)
        n_t = a.n_thought_tokens
        full_stream = torch.cat([m._build_thought_stream(sids), sids], dim=1)
        thought_mask = m._mask_cache.thought_causal(n_t, seq, x.device)
        thought_state = m.thought_tokens.init_state(B, x.device, x.dtype)
        prev_energy = None
        melt = {}
        if a.use_melt_kv:
            for bi, blk in enumerate(m.reasoning_blocks):
                if blk.use_melt:
                    melt[bi] = blk.melt_kv.init_state(B, n_t + seq, x.device, x.dtype)
        tr = [(x, None)] if trace else None
        for bi, t in cfg["plan"]:
            g = bi * a.max_reasoning_iters + t
            x_in = x
            thought_state, _, branches = m.thought_tokens.step_thought(thought_state, g, hard=(not m.training))
            x = m._apply_thought_bridge_real(thought_state, x, branches)
            x_with = m.thought_tokens.inject(x, thought_state)
            x_with, gate, E, ms_new, _ = m.reasoning_blocks[bi](x_with, full_stream, thought_mask,
                                                                prev_energy, melt.get(bi))
            x, thought_state = m.thought_tokens.strip(x_with)
            if E is not None:
                prev_energy = E
            if ms_new is not None:
                melt[bi] = ms_new
            if trace:
                gt = gate
                if gt is not None and gt.dim() >= 2 and gt.shape[1] == n_t + seq:
                    gt = gt[:, n_t:]
                tr.append((x, gt))
            if self.dyn is not None:
                xi = x_in.float()
                self.dyn.append(((x.float() - xi).norm(dim=-1) / xi.norm(dim=-1).clamp_min(1e-6)).mean().item())
        return x, tr

    @torch.no_grad()
    def run(self, ids, cfg):
        x, _ = self._run(ids, cfg)
        logits = self.m.output(self.m.final_norm(x)).float()
        return logits[0], self.cap["z"][0], self.cap["h"][0]

    @torch.no_grad()
    def run_batch(self, ids, cfg):
        x, _ = self._run(ids, cfg)
        return self.m.output(self.m.final_norm(x)).float()

    @torch.no_grad()
    def readout(self, x):
        return self.m.output(self.m.final_norm(x)).float()

    @torch.no_grad()
    def check_forward(self, ids):
        """Full schedule must reproduce VERAPsi.forward."""
        ref = self.m(ids, self.stream_ids(ids)).float()[0]
        mine, _, _ = self.run(ids, self.configs()[0][1])
        return float((ref - mine).abs().max())
