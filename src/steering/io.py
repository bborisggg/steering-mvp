"""Config-hashed caching, seeding, and paths.

`run_or_load(config, fn, force=False)` is the single entry point for anything expensive. The
cache key must include model revision, hook, vector-split hash, corruption, denoiser config,
generation config, seed, and metric version -- never a filename built from one or two
human-readable fields.
"""
