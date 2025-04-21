import base64
from clients.openai_client import OpenAIClient

class Captioner:
    def __init__(self, openai_client: OpenAIClient):
        self._ai = openai_client

    def caption_image(self, image_path: str, prev_captions: list[str] = []) -> str:
        prompt = self._generate_prompt(prev_captions[0] if len(prev_captions) > 0 else None)
        encoded_image = self.encode_image(image_path)

        return self._get_caption(prompt, encoded_image)

    def encode_image(self, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _generate_prompt(self, prev_caption: str | None):
        prompt = \
        f"""
        You are analyzing a single frame from a video.
        {f"This is the caption for the previous frame: {prev_caption}" if prev_caption else "This is the first frame in the video."}

        Describe exactly what's happening in the scene to produce a caption for the frame. Use a neutral, descriptive tone.

        - Include visible people, objects, setting, and actions
        - Be concise
        - Do NOT ask follow-up questions, make suggestions, or respond conversationally
        - Only describe what's visually present
        """

        return prompt

    def _get_caption(self, prompt: str, image_url: str) -> str:
        return self._ai.text_with_image_prompt(prompt, image_url)



