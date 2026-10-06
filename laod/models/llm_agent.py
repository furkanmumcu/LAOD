"""Vision-language model wrappers: image in, free-text object list out.

Most of the roster goes through ``transformers``'s ``image-text-to-text``
pipeline with a chat-formatted message. Phi-3.5-vision does not -- it ships
custom modeling code and its own ``<|image_1|>`` markup -- so it gets its own
adapter behind the same interface.

Decoding defaults to **greedy**, unlike the original pipeline which left
sampling on. Sampling makes a run unreproducible: re-running the same model on
the same image yields a different label list and therefore different numbers.
Greedy costs a little diversity and buys an experiment that can be repeated.
Pass ``do_sample=True`` with a seed to go back to the original behaviour; either
way the choice is recorded in the run's ``config.json``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

from laod.config import hf_token
from laod.models.registry import LLMS, LLMSpec, PromptSpec

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 256


class BaseLLMAgent(ABC):
    """Generate a comma-separated object list for an image."""

    def __init__(self, spec: LLMSpec, *, device_map: str = "auto",
                 dtype: str = "bfloat16", max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 do_sample: bool = False, **generation: Any) -> None:
        self.spec = spec
        self.device_map = device_map
        self.dtype = dtype
        self.generation = {"max_new_tokens": max_new_tokens,
                           "do_sample": do_sample, **generation}
        if spec.gated and not hf_token():
            raise RuntimeError(
                f"{spec.model_id} is gated and no HF_TOKEN is set. Add it to .env "
                "(see .env.example) and accept the licence on the model page.")
        self._load()

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def generate(self, image, prompt: PromptSpec) -> str:
        """Return the model's reply verbatim, unparsed."""

    def generate_many(self, images: Sequence, prompt: PromptSpec) -> list[str]:
        """Default sequential fallback; adapters may override with batching."""
        return [self.generate(im, prompt) for im in images]

    @property
    def generation_params(self) -> dict[str, Any]:
        return {"dtype": self.dtype, **self.generation}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.spec.key})"


class PipelineAgent(BaseLLMAgent):
    """Chat-formatted VLMs: Gemma-3, Gemma-4, Qwen2.5-VL, Qwen3.5."""

    def _load(self) -> None:
        import torch
        from transformers import pipeline
        kwargs: dict[str, Any] = {
            "model": self.spec.model_id,
            "torch_dtype": getattr(torch, self.dtype),
        }
        # An explicit "cuda:N" pins the whole model to one GPU; "auto" shards it.
        if self.device_map == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device"] = self.device_map
        if self.spec.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if (tok := hf_token()):
            kwargs["token"] = tok
        kwargs.update(self.spec.pipeline_kwargs)
        self.pipe = pipeline("image-text-to-text", **kwargs)
        logger.info("loaded %s (%s)", self.spec.key, self.spec.model_id)

    def _messages(self, image, prompt: PromptSpec) -> list[dict]:
        return [
            {"role": "system",
             "content": [{"type": "text", "text": prompt.system}]},
            {"role": "user",
             "content": [{"type": "text", "text": prompt.user},
                         {"type": "image", "url": image}]},
        ]

    def generate(self, image, prompt: PromptSpec) -> str:
        # Generation parameters must go through generate_kwargs. Passed as bare
        # **kwargs the pipeline routes them to the processor instead, where they
        # are silently ignored -- so do_sample=False never reaches generate()
        # and every model runs with whatever its own config defaults to. Gemma-3
        # defaults to do_sample=True, top_k=64, top_p=0.95, which made the
        # "greedy" runs sample.
        out = self.pipe(text=self._messages(image, prompt),
                        generate_kwargs=dict(self.generation))
        return str(out[0]["generated_text"][-1]["content"]).strip()


class Phi3VisionAgent(BaseLLMAgent):
    """Phi-3.5-vision: custom modeling code and <|image_1|> markup."""

    @staticmethod
    def _patch_stale_cache_api() -> None:
        """Teach the modern cache the two attributes Phi-3.5-vision expects.

        The checkpoint ships its own modeling code written against an older
        transformers, where the cache exposed ``seen_tokens`` and
        ``get_max_length()``. Both were removed, so generation dies with
        AttributeError. The replacements are exact -- ``get_seq_length()``
        returns what ``seen_tokens`` did, and an unbounded dynamic cache has no
        maximum -- so this restores the old surface rather than approximating it.
        """
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None
        if not hasattr(DynamicCache, "get_usable_length"):
            # An unbounded cache can always use everything it holds, so the
            # old method reduced to get_seq_length() for DynamicCache.
            DynamicCache.get_usable_length = (
                lambda self, new_seq_length=None, layer_idx=0: self.get_seq_length(layer_idx))

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self._patch_stale_cache_api()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            self.spec.model_id, trust_remote_code=True, num_crops=4)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id, trust_remote_code=True,
            torch_dtype=getattr(torch, self.dtype), device_map=self.device_map,
            _attn_implementation="eager",
        ).eval()
        logger.info("loaded %s (%s)", self.spec.key, self.spec.model_id)

    def generate(self, image, prompt: PromptSpec) -> str:
        messages = [{"role": "user",
                     "content": f"<|image_1|>\n{prompt.system}\n{prompt.user}"}]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text, [image], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            ids = self.model.generate(
                **inputs, eos_token_id=self.processor.tokenizer.eos_token_id,
                **self.generation)
        new = ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


_ADAPTERS = {"pipeline": PipelineAgent, "phi3_vision": Phi3VisionAgent}


def build_llm_agent(key: str, **kwargs) -> BaseLLMAgent:
    """Instantiate a label-generating VLM by registry key."""
    if key not in LLMS:
        raise ValueError(f"unknown LLM {key!r}; have {sorted(LLMS)}")
    spec = LLMS[key]
    if not spec.supported:
        raise RuntimeError(f"{key} is marked unsupported: {spec.notes}")
    return _ADAPTERS[spec.adapter](spec, **kwargs)
