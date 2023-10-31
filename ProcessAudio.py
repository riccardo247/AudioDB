from pydub import AudioSegment
from pydub.silence import split_on_silence
from demucs import separate
import speech_recognition as sr
import subprocess
import tempfile
import sys
from multiprocessing import Pool, cpu_count
from retrying import retry
import os
import argparse
import whisper
import numpy as np
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import librosa
import openai
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

openai.api_key = config['openai']['api_key']
class EngineSteam():
    """

    """
    def __init__(self):
        #this is init for deep denoise
        self.model, self.df_state, _ = init_df()

    def speech_to_text(self, audio_path):
        """
        :param audio_path: file in input wav
        :return: speech text
        """
        r = sr.Recognizer()
        chunk = AudioSegment.from_mp3(audio_path)
        filename = self.get_filename(audio_path)
        temp_directory = tempfile.gettempdir()
        temp = os.path.join(temp_directory, f'{filename}_temp.wav')
        chunk.export(temp, format='wav')
        with sr.AudioFile(temp) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            #text = r.recognize_google(audio_data)
            text = r.recognize_whisper_api(audio_data)
            #text = r.recognize_sphinx(audio_data)

        return text

    def change_extension(self, file_path, new_extension):
        """

        :param file_path: inout file complete path or filename with ext
        :param new_extension: new ext
        :return: filename with new extension. e.g. chunk0.wav
        """
        # Split the filename into name and extension
        base_name = os.path.splitext(file_path)[0]
        # Return the base name with the new extension
        return f"{base_name}.{new_extension}"

    def get_filename(self, file_path):
        """
        get basename name
        :param filepath: inout file path
        :return: return base name
        """
        # Get the base name (filename with extension, without directory path)
        base_name = os.path.basename(file_path)
        # Split the base name into filename and extension, and return the filename part
        return os.path.splitext(base_name)[0]
    def convert_to_wav(self, input_path, output_path=None):
        """
        convert any audio format to wav with ffmpeg
        :param input_path: inout music file
        :param output_file: output path with ext
        :return:
        """
        try:
            command = [
                "ffmpeg",
                "-i", input_path,
                "-acodec", "pcm_s16le",
                "-ar", "48000",
                output_path
            ]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print(f"Error converting {input_path} to {output_path}")
        except FileNotFoundError:
            print("FFmpeg not found. Ensure it's installed and in your system's PATH.")
    def convert_to_mp3(self, input_path, output_path=None):
        """
        convert any audio into mp3 with ffmpeg
        :param input_path: input music file
        :param output_path: output mp3 file
        :return: none
        """
        try:
            command = [
                "ffmpeg",
                "-i", input_path,
                output_path
            ]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print(f"Error converting {input_path} to {output_path}")
        except FileNotFoundError:
            print("FFmpeg not found. Ensure it's installed and in your system's PATH.")

    def demucs_file(self, audio_path):
        """
        run demucs on file. it is supposed to load model once at first call
        :param audio_path: input path
        :return: none
        """
        separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx", audio_path])

    def sample_file(self, audio_path, file_path, out_path):
        """
        sample music file in n seconds chunks with AudioSegment. exclude silence segments
        :param audio_path: input audio path
        :param file_path: output file
        :param out_path: output path
        :return: none
        """
        # Load audio file
        sound = AudioSegment.from_mp3(audio_path)
        file_path = self.get_filename(file_path)
        dir_name = os.path.join(out_path, file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # Break into segments based on silence
        segments = split_on_silence(sound, silence_thresh=sound.dBFS - 30, min_silence_len=5000)

        # Loop through the segments to make 10-second chunks
        start_time = 0
        end_time = 10000  # 10 seconds in milliseconds
        print(f"found {len(segments)} segments")
        for n, segment in enumerate(segments):
            segment_length = len(segment)
            print(f"seg length {segment_length} ")
            while end_time < segment_length:
                # Export 10-second chunk
                chunk = segment[start_time:end_time]
                chunk_name = f"chunk_{n}_start_{start_time}_end_{end_time}.mp3"
                outname = os.path.join(dir_name, chunk_name)
                chunk.export(outname, format="mp3")
                # Update start and end times for the next chunk
                start_time += 10000
                end_time += 10000

            # Reset for next segment
            start_time = 0
            end_time = 10000

    def get_pitches(self, audio_path, pitch_path):
        labels = ["main_pitch", "power_signal", "power_noise", "snr"]
        print(f"processing for pitches: {audio_path}")
        # Ensure txtdir exists
        if not pitch_path:
            pitch_path = audio_path
        if not os.path.exists(pitch_path):
            os.makedirs(pitch_path)

        # Iterate over each file in dirpath
        for filename in os.listdir(audio_path):
            filepath = os.path.join(audio_path, filename)
            print(f"file: {filename}")
            try:
                pitch_data = self.get_pitch(filepath)
            except Exception as e:
                print(f" error {e} in pitch processing {filename}")
                pitch_data = []
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_filepath = os.path.join(pitch_path, output_filename)
            with open(output_filepath, 'w', encoding='utf-8') as output_file:
                for label, number in zip(labels, pitch_data):
                    formatted_number = f"{label}:{number:.3f}"
                    output_file.write(formatted_number + '\n')
    def get_pitch(self, filepath):

        y, sr = librosa.load(filepath)
        # Estimate the constant-Q chromagram
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # Select out pitches with high energy
        index = magnitudes.argmax(axis=0)
        pitch = pitches[index, np.arange(pitches.shape[1])]

        # Filter out zero values (unvoiced frames)
        pitch = pitch[pitch > 0]

        # Calculate the mean pitch
        main_pitch = np.mean(pitch)
        threshold = 200 #hz
        signal = pitches[(pitches > main_pitch - threshold) & (pitches < main_pitch + threshold)]

        # Isolate the noise
        noise = pitches[(pitches < 300) | (pitches > 10000)]  #

        # Compute the power of signal and noise
        power_signal = np.mean(signal ** 2)
        power_noise = np.mean(noise ** 2)

        # Calculate SNR
        snr = 10 * np.log10(power_signal / power_noise)
        print(f"main pitch is {main_pitch}, signal power:{power_signal}, noise power:{power_noise}, snr:{snr}")
        return [main_pitch, power_signal, power_noise, snr]
    def deep_denoise(self, audio_path,  filename_in=None, out_path=None, denoise=True):
        """
        apply deep denoise filter with deep denoise
        :param filename_in:
        :param audio_path: input audio mp3 file or directory containing wav audio files
        :param out_path: output path or none if single file
        :param denoise: if run denoise or only converto to wav if necessary
        :return: none
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if os.path.isdir(audio_path):
            #should be wav files
            file_list = os.listdir(audio_path)
        else:
            #should be mp3
            audio_path, audio_file = os.path.split(audio_path)
            file_list = [audio_file]
        #from here on
        #filelist must contain filenames not full path
        #audio_pat must be a path
        for filename in file_list:
            filepath = os.path.join(audio_path, filename)
            _, extension = os.path.splitext(filename)
            if filename_in is not None:
                fileout = os.path.join(out_path, filename_in+".wav")
            else:
                #TODO not tested
                fileout = os.path.join(out_path, self.change_extension(filename, "wav"))
            print(f"inside denoise,  convert {filepath} to {fileout}")
            self.convert_to_wav(filepath, fileout)
            if denoise:
                print(f"denoise {fileout}")
                #works on wav
                audio, _ = load_audio(fileout, sr=self.df_state.sr())
                # Denoise the audio
                enhanced = enhance(self.model, self.df_state, audio)
                # Save for listening
                save_audio(fileout, enhanced, self.df_state.sr())

    def get_voice_description(self, audio_path):
        """
        TODO can be done in a second step in another virtual env because libraries versions are different
        :param audio_path:
        :return:
        """
        pass
    def get_texts(self, audio_path, txt_path=None):
        """
        get text from audio using either whisper(best ) or google
        whisper nees openai keys
        :param audio_path: audio file in
        :param txt_path: path of txt output
        :return:
        """
        print(f"processing: {audio_path}")
        # Ensure txtdir exists
        if not txt_path:
            txt_path = audio_path
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)

        # Iterate over each file in dirpath
        for filename in os.listdir(audio_path):
            filepath = os.path.join(audio_path, filename)
            print(f"file: {filename}")
            try:
                text = self.speech_to_text(filepath)
            except Exception as e:
                print(f"error: {e} ")
                text = ""
            # Save processed content in txtdir with .txt extension
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_filepath = os.path.join(txt_path, output_filename)
            with open(output_filepath, 'w', encoding='utf-8') as output_file:
                output_file.write(text)

    def process_all(self, file_path, out_path, skip_n=0):
        """
        process all chain for all files in file_path
        :param file_path: path of dir in input
        :param out_path: output dir for all files
        :return:
        """
        for index, filename in enumerate(os.listdir(file_path)):
            if index < skip_n:
                continue  # Skip the first n files
            filepath = os.path.join(file_path, filename)
            try:
                self.process_file(filepath, out_path)
            except Exception as e:
                print(f"error in processing file: {e}")

    def process_file(self, file_path, out_path):
        """
        process a single file : file_path
        steps:
        -convert to mp3
        -demucs into voice
        -denoise vocals
        -sample into n sec chunks excluding silence
        -denoise each chunk?
         it is creating directories into out_path, using output dir of demucs and tmp dir . check total size
         important files are chunks_clean and txt output
        :param file_path: input file
        :param out_path: output path
        :return:
        """

        cleandir = os.path.join(out_path, r"voice_clean")
        mp3dir = os.path.join(out_path, r"mp3")
        chunkdir = os.path.join(out_path, self.get_filename(file_path))
        cleachunkndir = os.path.join(out_path, self.get_filename(file_path) + "_clean")
        txtdir = os.path.join(out_path, self.get_filename(file_path) + "_txt")
        pitchdir = os.path.join(out_path, self.get_filename(file_path) + "_pitch")
        if not os.path.exists(mp3dir):
            os.makedirs(mp3dir)
        #convert to mp3
        fileout = self.change_extension(self.get_filename(file_path), "mp3")
        fileout = os.path.join(mp3dir, fileout)
        self.convert_to_mp3(file_path, fileout)
        #demucs files
        self.demucs_file(fileout)
        #get audio vocals from demucs
        file_path_ext = self.get_filename(file_path) + ".wav"
        file_path = self.get_filename(file_path)
        voice_file = os.path.join(os.getcwd(), "separated", "mdx", file_path, "vocals.mp3")
        #denoise before sampling.  mp3 to wav into cleandir with same name
        self.deep_denoise(voice_file, file_path, cleandir, denoise=True)
        input_wav = os.path.join(cleandir, file_path + ".wav")
        output_mp3 = os.path.join(cleandir, file_path + ".mp3")
        #go back to mp3.
        print(f"{input_wav} to {output_mp3}")
        self.convert_to_mp3(input_wav, output_mp3)
        # sample file
        self.sample_file(output_mp3, file_path, out_path)
        ##denoise each sample?
        ##denoise=True not tested here do not use by now
        ##self.deep_denoise(chunkdir, cleachunkndir, denoise=False)
        # for each segment speech to text
        self.get_texts(chunkdir, txtdir)
        self.get_pitches(chunkdir, pitchdir)



def main():
    parser = argparse.ArgumentParser(description="Process audio files in chunks and get tedt, pitch and voice description")
    parser.add_argument("path_to_directory", help="directory containing input files")
    parser.add_argument("output_dir", help="directory output files")
    parser.add_argument("--skipn", help="Optional skip first n files", default=0)
    parser.add_argument("--loadn", help="process n files", default=100)
    args = parser.parse_args()
    if len(sys.argv) < 3:
        print("Usage: python script.py <path_to_directory> <output_dir> --skipn <number of files to skip>")
        sys.exit(1)

    dir_path = args.path_to_directory
    out_path = args.output_dir
    skipn = args.skipn
    loadn = args.loadn
    engine = EngineSteam()
    engine.process_all(dir_path, out_path, skipn, loadn)
    #engine.process_all(r"C:\Users\ricca\Downloads\audiofiles", r"C:\Users\ricca\Downloads\track1")
    #extract.process_file(r"C:\Users\ricca\Downloads\-0skjm-uJSs.ogg")


if __name__ == '__main__':
    main()