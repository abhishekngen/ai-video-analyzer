import subprocess
from pathlib import Path

class FrameExtractor:
    def __init__(self, fps=1):
        self.fps = fps

    def extract(self, video_path: str, overwrite=False) -> str:
        frames_dir = Path(f"./frames/{Path(video_path).parent.stem}")
        frames_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite and any(frames_dir.glob("*.jpg")):
            print(f"Frames already extracted in {frames_dir}. Use overwrite=True to extract again.")
            return str(frames_dir)

        output_pattern = str(frames_dir / "frame_%04d.jpg")
        subprocess.run([
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={self.fps}",
            output_pattern
        ], check=True)

        return str(frames_dir)
