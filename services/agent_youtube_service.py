import base64

from pytubefix import YouTube
from pytubefix import Search
from pytubefix.helpers import safe_filename

from services.agent_chroma_db import ImageChromaDB
from services.frame_extractor import FrameExtractor

class Youtube_Service:
    def __init__(self):
        self.chroma = ImageChromaDB()

    def search_video(self, search_query: str):
        results = Search(search_query)
        video_results = list(map(lambda v: {'title': v.title, 'url': v.watch_url}, results.videos))
        return str(video_results)

    def get_video_captions(self, url: str):
        yt = YouTube(url)
        captions = yt.captions

        english_captions = captions['a.en'].generate_srt_captions() if 'a.en' in captions else 'No english captions available'

        return str(english_captions)

    def download_and_index_video(self, url: str):
        try:
            yt = YouTube(url)
        except Exception as e:
            return f"URL provided was not valid, you should verify the URL with the search function."
        id = yt.video_id
        if yt.age_restricted:
            return "Video is unfortunately age restricted and cannot be downloaded."
        video_dir = f"./video_downloads/{id}"
        print('Downloading video to:', video_dir)
        ys = yt.streams.get_highest_resolution()
        downloaded_video_dir = ys.download(output_path=video_dir) # Could be None
        print('Downloaded video to:', downloaded_video_dir)

        frame_extractor = FrameExtractor(fps=2)
        frames_dir = frame_extractor.extract(downloaded_video_dir, False)
        print('Extracted frames to:', frames_dir)

        self.chroma.create_collection(id)

        return "Video frames indexed to collection: " + id + ". The video has duration " + str(yt.length) + " seconds. You can now query the frames using the `query_embedding` function."

    def index_video(self, downloaded_video_id: str):
        downloaded_video_dir = f"./video_downloads/{downloaded_video_id}/{downloaded_video_id}.mp4"
        frame_extractor = FrameExtractor(fps=2)
        frames_dir = frame_extractor.extract(downloaded_video_dir, False)
        print('Extracted frames to:', frames_dir)

        self.chroma.create_collection(downloaded_video_id)

        return "Video frames indexed to collection: " + downloaded_video_id + ". You can now query the frames using the `query_embedding` function."

    def get_video_id(self, url: str):
        yt = YouTube(url)
        return yt.video_id

    def get_video_frames(self, video_id: str, seconds_timestamp: int):
        uri_1 = f"./frames/{video_id}/frame_{(seconds_timestamp*2):04d}.jpg"
        uri_2 = f"./frames/{video_id}/frame_{(seconds_timestamp * 2 + 1):04d}.jpg"
        image_1 = base64.b64encode(open(uri_1, 'rb').read()).decode('utf-8')
        image_2 = base64.b64encode(open(uri_2, 'rb').read()).decode('utf-8')

        return [{ "type": "input_image",
          "image_url": f"data:image/jpeg;base64,{image_1}" },
            {"type": "input_image",
             "image_url": f"data:image/jpeg;base64,{image_2}"},
                {
                    "type": "input_text", "text": "This is not a prompt from the user, but just a delivery of the two frames you requested. Note that the user cannot see this message nor the images."
                },
        ]
