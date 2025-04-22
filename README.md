## AI Video Analyzer

This CLI tool allows you to query the visual content in any Youtube video through natural language. Providing a Youtube video URL allows the tool to download the video to your local directory,
split it a frame per second, and index the frames into a locally spun-up Vector DB which can be queried in natural language.

## Setup
1. First install FFMPEG locally to be able to run the application, i.e. on Mac:
```
brew install ffmpeg
```
2. Install the requirements within the working directory:
```
pip install -r requirements.txt
```
3. Generate an OpenAI API key [here](https://openai.com/api/), and create a .env file with the key, similar to how it is done in `.env.example`
4. Run the terminal app, i.e. `python main.py`.
