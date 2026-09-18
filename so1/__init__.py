"""Open Alternative to Jev (import name so1, "System One"): typed, calibrated decisions from any open-weights LLM in a single forward pass.

    from so1 import Decider, Choice
    decider = Decider.from_pretrained("Qwen/Qwen3.6-27B", backend="hf", load_in_8bit=True)
    decisions = decider.decide(
        state="Customer email: ...",
        questions=[Choice("Is this a refund request?", ["yes", "no"]),
                   Choice("Urgency", ["low", "medium", "high"])],
    )
    decisions[0].choice, decisions[0].confidence
"""
from .schema import Choice, Decision, yes_no, scale
from .prompting import ChatFormat, PromptBuilder, PackedPrompt
from .decider import Decider
from .calibration import TemperatureScaler, expected_calibration_error, cross_fit_temperature

__version__ = "0.1.0"
__all__ = ["Choice", "Decision", "yes_no", "scale", "ChatFormat", "PromptBuilder", "PackedPrompt", "Decider",
           "TemperatureScaler", "expected_calibration_error", "cross_fit_temperature", "__version__"]
