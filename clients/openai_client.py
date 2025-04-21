from dotenv import load_dotenv
from openai import OpenAI


class OpenAIClient:
    def __init__(self):
        load_dotenv()

        self._client = OpenAI()

    def embed_text(self, text: str):
        response = self._client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

        return response.data[0].embedding

    def text_prompt(self, prompt: str):
        response = self._client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text", "text": prompt
                        }
                    ]
                }
            ]
        )

        return response.output_text

    def text_with_image_prompt(self, prompt: str, image_url: str):
        response = self._client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text", "text": prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_url}",
                        },
                    ],
                }
            ],
        )

        return response.output_text
