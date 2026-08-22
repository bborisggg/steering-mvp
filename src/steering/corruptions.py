"""What the denoiser trains on: gaussian, variance_preserving, rank1.

`rank1` takes a configurable direction pool -- pool size is a parameter, not a separate class.
Each family reports the conditioning signal it was trained with (see DECISIONS.md #3).
"""
