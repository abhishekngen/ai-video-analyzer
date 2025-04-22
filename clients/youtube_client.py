from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix.helpers import safe_filename
from pathlib import Path

class YoutubeClient:
    def download_video(self, url: str, overwrite: bool = False) -> (str, str, str):
        yt = YouTube(url, on_progress_callback=on_progress)
        video_dir = Path(f"./video_downloads/{safe_filename(yt.title)}_{yt.video_id}")
        if not overwrite and video_dir.exists():
            mp4_files = list(video_dir.glob("*.mp4"))
            if len(mp4_files) >= 1:
                print(f"Video already downloaded in {mp4_files[0]}. Use overwrite=True to download again.")
                return str(mp4_files[0]), yt.title, yt.video_id

        ys = yt.streams.get_highest_resolution()
        downloaded_video_dir = ys.download(output_path=str(video_dir))
        return downloaded_video_dir, yt.title, yt.video_id
