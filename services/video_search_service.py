from clients.chroma_client import ChromaClient
from services.frame_extractor import FrameExtractor
from services.frame_indexer import FrameIndexer
from clients.openai_client import OpenAIClient
from clients.youtube_client import YoutubeClient


class VideoSearchService:
    def __init__(self, openai_client: OpenAIClient, chroma_client: ChromaClient, youtube_client: YoutubeClient, collection_name: str = "video_frames"):
        self.ai = openai_client
        self.chroma = chroma_client
        self.youtube = youtube_client

    def run_download_and_index_video(self):
        video_url = input("Enter the YouTube video URL: ")
        print("Downloading video...")
        video_path = self.youtube.download_video(video_url)
        print(f"Video downloaded to {video_path}.")
        #
        fps = 1  # Extract one frame per second
        #     frames_dir = extract_frames(video_path, fps=fps, overwrite=True)
        #     print(f"Frames extracted to {frames_dir}.")
        print("Extracting frames...")

        frame_extractor = FrameExtractor(fps=fps)
        frames_dir = frame_extractor.extract(video_path, False)

        print(f"Frames extracted to {frames_dir}")

        print("Indexing frames...")

        frame_indexer = FrameIndexer(self.ai, self.chroma, frames_dir, fps)
        frame_indexer.index_frames()

        print("Video frames indexed.")

    def run_interactive_loop(self):
        while True:
            query = input("Enter your query (or 'exit' to quit): ")
            if query.lower() == "exit":
                break

            augmented_query = self.augment_search_query(query)
            use_augmented = input(f"Use this augmented query instead? Y/n: {augmented_query} ")
            if use_augmented.strip().lower() == "y":
                query = augmented_query

            query_embedding = self.ai.embed_text(query)

            results = self.chroma.query_embedding(query_embedding, 5, ["documents", "metadatas"])

            context_str = self.generate_context_string(results)

            final_response = self.generate_search_response(query, context_str)
            print("\nFinal Response:\n" + final_response)

    def augment_search_query(self, query: str):
        return self.ai.text_prompt(
            f"""
            Rephrase the following video search query as a direct image caption that could be used in a cosine similarity search, assuming the video has been split into frames that have been captioned and embedded. Match the tone and style of image descriptions.
            Return exactly the rephrased caption, no extra text.
            {query}
            """
        )

    def generate_context_string(self, results):
        return "\n".join(
            f"[{meta['timestamp_sec']}s] {doc}"
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        )

    def generate_search_response(self, query: str, context_str: str):
        return self.ai.text_prompt(
            f"""
            You are helping analyze a video.
            Below are the top matching frame captions and their timestamps for a user query.

            Query: {query}

            Search Results:
            {context_str}

            Generate a concise response to the user query using the search results. Include timestamps for each result.
            """
        )
