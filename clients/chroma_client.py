import chromadb

class ChromaClient:
    def __init__(self, collection_name: str = "video_frames"):
        self._chroma_client = chromadb.Client()
        self._collection = self._chroma_client.get_or_create_collection(collection_name)

    def add_item_to_collection(self, id: str, embedding: list, item: str, metadata: dict):
        self._collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[item],
            metadatas=[metadata]
        )

    def query_embedding(self, query_embedding: list, n_results: int = 5, include: list = None):
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include
        )

        return results


