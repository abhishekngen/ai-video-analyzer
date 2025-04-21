from clients.chroma_client import ChromaClient
from clients.openai_client import OpenAIClient
from services.video_search_service import VideoSearchService
from clients.youtube_client import YoutubeClient

if __name__ == "__main__":
    client = OpenAIClient()
    chroma_client = ChromaClient()
    youtube_client = YoutubeClient()
    video_search_service = VideoSearchService(client, chroma_client, youtube_client)

    video_search_service.run_download_and_index_video()
    video_search_service.run_interactive_loop()
