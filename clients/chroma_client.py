import chromadb

class ChromaClient:
    def __init__(self):
        self._chroma_client = chromadb.PersistentClient()

    def get_collection(self, collection_name: str):
        try:
            return self._chroma_client.get_collection(name=collection_name)
        except:
            return None

    def get_collections(self):
        return self._chroma_client.list_collections()

    def create_collection(self, collection_name: str):
        return self._chroma_client.get_or_create_collection(name=collection_name)

    def add_item_to_collection(self, collection_name: str, item_id: str, embedding: list, item: str, metadata: dict):
        self.get_collection(collection_name).add(
            ids=[item_id],
            embeddings=[embedding],
            documents=[item],
            metadatas=[metadata]
        )

    def query_embedding(self, collection_name: str, query_embedding: list, n_results: int = 5, include: list = None):
        results = self.get_collection(collection_name).query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include
        )

        return results


