from pathlib import Path

from tqdm import tqdm

from services.captioner import Captioner
from clients.chroma_client import ChromaClient
from clients.openai_client import OpenAIClient


class FrameIndexer:
    def __init__(self, openai_client: OpenAIClient, chroma_client: ChromaClient, frames_dir: str, fps: int = 1):
        self._ai = openai_client
        self._vector_db = chroma_client
        self._captioner = Captioner(openai_client)
        self.frames_dir = frames_dir
        self.fps = fps

    def index_frames(self):
        captions = []
        for frame_path in tqdm(Path(self.frames_dir).glob("*.jpg"), desc="Indexing frames"):
            caption = self._captioner.caption_image(frame_path, captions)
            self.index_frame(frame_path, caption)
            captions = [caption]

    def index_frame(self, frame_path: str, caption: str):
        frame_id = Path(frame_path).stem
        frame_number = int(frame_id.split("_")[-1])
        timestamp_sec = (frame_number - 1) // self.fps

        frame_embedding = self._ai.embed_text(caption)

        metadata = {
            "timestamp_sec": timestamp_sec,
        }

        self._vector_db.add_item_to_collection(
            id=frame_id,
            embedding=frame_embedding,
            item=caption,
            metadata=metadata
        )
