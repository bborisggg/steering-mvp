"""The residual MLP: D(x, s) = x + f(N(x), e(s)), output head zero-initialised.

Zero-init means an untrained denoiser is exactly the identity, which is the correct prior and a
test.
"""
