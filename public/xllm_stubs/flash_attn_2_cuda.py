def __getattr__(name):
    raise NotImplementedError("flash attention is not available; use causal_attn_backend=None")
