from .nodes.ASVPromptGenerator import ASVPromptGenerator


NODE_CLASS_MAPPINGS = {"ASVPromptGenerator": ASVPromptGenerator}

NODE_DISPLAY_NAME_MAPPINGS = {"ASVPromptGenerator": "ASV Prompt Generator"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
