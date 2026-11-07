"""lowrankbandit: greedy low-rank contextual GLM bandits and two-track inference for linear functionals.

    GreedyPolicy         Algorithm 1: uniform exploration, then greedy choices from a nuclear-norm-penalized GLM fit
    TwoTrackPolicy       Step 1 of Algorithm 2: shared exploration, then two greedy tracks in turn
    two_track_inference  Steps 2-4 of Algorithm 2: confidence intervals for <A, theta*>
    fit_nuclear          the nuclear-norm-penalized GLM estimator on given rows
    Stream, simulate_stream, run, low_rank_truth   simulated data for examples
"""
from .model import interaction_loading, vec
from .estimator import fit_nuclear
from .policy import GreedyPolicy, TwoTrackPolicy
from .inference import two_track_inference
from .simulate import Stream, simulate_stream, run, low_rank_truth

__version__ = "0.1.0"
__all__ = ["GreedyPolicy", "TwoTrackPolicy", "two_track_inference", "fit_nuclear", "interaction_loading", "vec",
           "Stream", "simulate_stream", "run", "low_rank_truth"]
