"""tollama-eval: automated time series benchmarking."""

__version__ = "0.2.0"


def __getattr__(name: str) -> object:
    """Lazy import for TSAutopilot to avoid heavy deps on import."""
    if name == "TSAutopilot":
        from ts_autopilot.sdk import TSAutopilot

        return TSAutopilot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
