import base64
from PIL import Image

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

class ImageChromaDB:
    def __init__(self):
        self._chroma_client = chromadb.PersistentClient(path='./agent_chroma')
        self.embedding_function = OpenCLIPEmbeddingFunction()
        self.image_loader = ImageLoader()

    def get_collection(self, video_id: str):
        try:
            return self._chroma_client.get_collection(name=video_id)
        except:
            return None

    def create_collection(self, video_id: str):
        if self.get_collection(video_id):
            # self._chroma_client.delete_collection(video_id)
            return video_id

        collection = self._chroma_client.create_collection(
            name=video_id,
            embedding_function=self.embedding_function,
            data_loader=self.image_loader)

        image_folder = f"./frames/{video_id}"

        image_uris = sorted([os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)])
        ids = [str(i) for i in range(len(image_uris))]

        for i in range(len(image_uris)):
            collection.add(ids=[str(i)], uris=[image_uris[i]], metadatas=[{"timestamp_sec": i // 2}])

        return video_id

    def query_embedding(self, video_id: str, search_query: str, n_results: int = 8, include=None):
        if include is None:
            include = ["data", "metadatas"]
        collection = self.get_collection(video_id)
        if collection:
            results = collection.query(
                query_texts=[search_query],
                n_results=n_results,
                include=include
            )
            for uri in results['uris'][0]:
                width = 80
                img = Image.open(uri)
                img = img.convert('L')  # grayscale
                aspect_ratio = img.height / img.width
                new_height = int(aspect_ratio * width * 0.55)
                img = img.resize((width, new_height))

                chars = " .:-=+*#%@"
                pixels = img.getdata()
                ascii_str = "".join(chars[pixel * (len(chars) - 1) // 255] for pixel in pixels)

                # for i in range(0, len(ascii_str), width):
                #     print(ascii_str[i:i + width])
            return ([{"type": "input_image",
                      "image_url": f"data:image/jpeg;base64,{base64.b64encode(open(uri, 'rb').read()).decode('utf-8')}"}
                     for uri in results['uris'][0]], results['metadatas'][0])

        return "Collection/results not found. You may need to download and index the video first."

    def get_collections(self):
        return str(self._chroma_client.list_collections())

