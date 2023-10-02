import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
from demucs import separate
import speech_recognition as sr
import sys
from pytube import YouTube
from moviepy.editor import *
from multiprocessing import Pool, cpu_count
from IPython.display import Audio
from retrying import retry


class Audio_db():
    def __init__(self, songs_dir):
        self.songs_dir = songs_dir

    #def load_songs_db(self):
    #    self.songs_df = pd.read_parquet('data/train-00000-of-00002-efd5ead509f2d961.parquet', engine='pyarrow')
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def get_audio(self, index, youtube_url, output_filename):
        if output_filename is None:
          output_filename = os.path.join(self.songs_dir,f"audio_{index}__"+youtube_url.split("=")[1]+".mp3")
        #Download video from YouTube
        yt = YouTube(youtube_url)
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        video_filename = video_stream.download()

        # Extract audio using moviepy
        video = VideoFileClip(video_filename)
        audio = video.audio
        audio.write_audiofile(output_filename)

        # Clean up the video file
        video.reader.close()
        video.audio.reader.close_proc()
        os.remove(video_filename)
        return f"Audio extracted to: {output_filename}"
    def youtube_getaudio(self, index, youtube_url, output_filename=None):
        try:
          message = self.get_audio(index, youtube_url, output_filename)
        except Exception as e:
          return f"Oops! failed: {index}, {youtube_url}, {str(e)}"
        return message

def main():
    #!git clone https://huggingface.co/datasets/DISCOX/DISCO-200K-high-quality
    os.chdir("~/DISCO-200K-high-quality")

    df = pd.read_parquet('data/train-00000-of-00002-efd5ead509f2d961.parquet', engine='pyarrow')
    audio_db = Audio_db("~/Audio_DB/mp3")  # /content/drive/MyDrive/Audio_DB/mp3
    def worker(data):
        index, row = data
        return audio_db.youtube_getaudio(index, row['video_url_youtube'])

    num_processes = cpu_count()
    print(f" num processes: {num_processes}")

    with Pool(processes= num_processes) as pool:
        for result in pool.map(worker, df.iterrows(), 1):
            print(result)
if __name__ == '__main__':
    main()