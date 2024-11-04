import subprocess
import sys
import random
import json
import os
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("roborovski/superprompt-v1", legacy=False)
model = T5ForConditionalGeneration.from_pretrained(
    "roborovski/superprompt-v1", device_map="auto"
)


def setup_package(pkg_name):
    try:
        __import__(pkg_name)
    except ImportError:
        print(f"{pkg_name} not found. Commencing installation...")
        subprocess.call([sys.executable, "-m", "pip", "install", pkg_name])
    finally:
        globals()[pkg_name] = __import__(pkg_name)


class TextGenerator:
    def __init__(self, initial_seed=None):
        self.randomizer = random.Random(initial_seed)

    def sanitize_string(self, dirty_string):
        return re.sub(r",+", ",", dirty_string.strip())

    def format_prompt(self, raw_text, initial_seed):
        return raw_text, initial_seed, "", "", ""

    def create_prompt(self, initial_seed, enabled, prompt, sampling, temperature, top_k, top_p, notes):

        kwargs = locals()
        del kwargs["self"]

        seed = kwargs.get("seed", 0)
        if seed is not None:
            self.randomizer = random.Random(seed)

        
        prompt = kwargs.get("prompt", "")
        
        if enabled: 
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(
                input_ids,
                max_new_tokens=1000,
                do_sample=sampling,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.format_prompt(self.sanitize_string(generated_text), initial_seed)
        else:
            return self.format_prompt(self.sanitize_string(prompt), initial_seed)

class ASVPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": random.randint(0, 999999999999999),
                        "min": 0,
                        "max": 999999999999999,
                        "step": 1,
                    },
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "sampling": ("BOOLEAN", {"default": True}),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0,
                        "max": 1.5,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0,
                        "max": 1,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
                "notes": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "README:\n\n* Sampling: Enables random sampling, allowing the model to generate more varied and creative outputs by choosing from a range of probable tokens instead of only the most likely one.\n* Temperature: Controls the randomness of predictions by scaling the logits before applying softmax. Higher values (e.g., 0.8 to 1.5) result in more randomness, while values closer to 0 make the output more deterministic.\n* Top_k: Limits the sample pool to the top k highest-probability next tokens, adding randomness by only sampling from this subset.\n* Top_p: Implements 'nucleus sampling', which includes tokens with a cumulative probability above a threshold (e.g., 0.9), allowing for dynamic vocabulary selection based on the context.\n\nImportant: This textbox serves as a README and a personal note-taking space. It is solely for reference and will not impact image generation in any way.\n\nhttps://github.com/zubenelakrab/ComfyUI-ASV-Nodes",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Prompt"

    def execute(self, seed, enabled, prompt, sampling, temperature, top_k, top_p, notes):
        generator = TextGenerator(seed)
        prompt = generator.create_prompt(
            seed, enabled, prompt, sampling, temperature, top_k, top_p, notes
        )
        return prompt

    @classmethod
    def HAS_CHANGED(cls, *args):
        return True


NODE_DEFINITIONS = {"ASVPromptGenerator": ASVPromptGenerator}
DISPLAY_NAME_MAP = {"ASVPromptGenerator": "ASV Prompt Generator"}
