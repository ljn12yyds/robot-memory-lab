from __future__ import annotations

from pathlib import Path


class LocalQwen3VLGenerator:
    """Callable Qwen3-VL backend for StructuredQwenPlanner."""

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 512,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()

    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str | None = None,
    ) -> str:
        content: list[dict[str, str]] = []
        if image_path:
            image = Path(image_path).expanduser().resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            content.append({"type": "image", "image": str(image)})
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        trimmed = [
            output[len(prompt) :]
            for prompt, output in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
