"""
Compatibility shim for extensions expecting AUTOMATIC1111's modules.sd_hijack.
Provides model_hijack.get_prompt_lengths used by sd-webui-prompt-all-in-one.
"""

from typing import Tuple


def _get_prompt_lengths(prompt: str, cond_stage_model=None) -> Tuple[int, int]:
    try:
        # Prefer the model-provided helper if available (Forge engines implement this).
        if cond_stage_model and hasattr(cond_stage_model, "get_prompt_lengths_on_ui"):
            return cond_stage_model.get_prompt_lengths_on_ui(prompt)

        from modules import sd_models  # type: ignore

        sd_model = getattr(sd_models, "model_data", None)
        sd_model = getattr(sd_model, "sd_model", None)
        fn = getattr(sd_model, "get_prompt_lengths_on_ui", None)
        if fn:
            return fn(prompt)
    except Exception:
        pass

    # Fallback: unknown length
    return 0, 0


class _ModelHijack:
    @staticmethod
    def get_prompt_lengths(prompt: str, cond_stage_model=None) -> Tuple[int, int]:
        return _get_prompt_lengths(prompt, cond_stage_model)


model_hijack = _ModelHijack()
