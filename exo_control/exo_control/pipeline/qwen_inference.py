import time
from typing import Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROMPT_JSON = """You will see an image of a single object (cropped ROI).

Return STRICT JSON only. No extra text.

{
  "object_name": "<1-3 words, English>",
  "weight_level": "LOW|MEDIUM|HEAVY",
  "fragility_level": "LOW|MEDIUM|HIGH"
}

Weight level definitions:
- LOW: easy to hold with one hand for a long time
- MEDIUM: can be held with one hand but causes fatigue
- HEAVY: difficult with one hand; likely needs two hands

Fragility risk definitions:
- LOW: unlikely to break if dropped
- MEDIUM: may be damaged if dropped
- HIGH: easily cracks/shatters/malfunctions if dropped

If uncertain, choose the closest level. Use ONLY the allowed level strings.
"""


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class Qwen2_5VLBackend:
    """
   Qwen2.5-VL backend for the perception pipeline.
    """

    def __init__(self, model_id: str, fp16: bool = True):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()

    def warmup(self, img: Image.Image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Hello"},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(images=img, text=text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )

        sync_if_cuda(device)

    def infer(
        self,
        img: Image.Image,
        prompt: str = PROMPT_JSON,
        max_new_tokens: int = 32,
    ) -> Tuple[str, float, float, float, float]:
        """
        Returns:
            out_text, vlm_ms, processor_ms, generate_ms, decode_ms
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        device = next(self.model.parameters()).device

        t_p0 = now_ms()
        inputs = self.processor(images=img, text=text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        processor_ms = now_ms() - t_p0

        sync_if_cuda(device)
        t_g0 = now_ms()
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        sync_if_cuda(device)
        generate_ms = now_ms() - t_g0

        t_d0 = now_ms()
        prompt_len = inputs["input_ids"].shape[-1]
        gen_only = out_ids[0][prompt_len:]
        out_text = self.processor.decode(
            gen_only,
            skip_special_tokens=True,
        ).strip()
        decode_ms = now_ms() - t_d0

        vlm_ms = processor_ms + generate_ms + decode_ms
        return out_text, vlm_ms, processor_ms, generate_ms, decode_ms
