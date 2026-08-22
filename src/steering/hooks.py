"""Residual-stream read/write at a frozen layer.

Owns the layer convention (`resid_post` of block L == `resid_pre` of block L+1) and guarantees
the intervention fires on every autoregressive forward, not only on the prompt.
"""
