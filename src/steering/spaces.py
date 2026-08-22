"""The one activation space corruptions and the denoiser live in.

`encode(h) -> z` and `decode(z) -> h`. Every centering, scaling, or normalisation belongs here;
no hand-written scale corrections anywhere else. `decode(encode(h)) ~= h` is a test.
"""
