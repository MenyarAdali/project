import os
from pydub import AudioSegment
import ffmpeg
# Set the path to ffmpeg for pydub
#AudioSegment.ffmpeg = ffmpeg_path

# Function to convert mp3 to wav
def convert_to_wav(mp3_file, output_folder):
    sound = AudioSegment.from_mp3(mp3_file)
    wav_filename = os.path.splitext(os.path.basename(mp3_file))[0] + '.wav'
    wav_path = os.path.join(output_folder, wav_filename)
    sound.export(wav_path, format="wav")

# Directory containing your mp3 files
mp3_directory = r"C:\Users\LENOVO\Desktop\speech_recognition\MLTU\Datasets\cv-valid-train"

# Directory to save converted wav files
output_directory = r"C:\Users\LENOVO\Desktop\speech_recognition\MLTU\Datasets\output_directory"

# Iterate through mp3 files and convert each one to wav
for filename in os.listdir(mp3_directory):
    if filename.endswith(".mp3"):
        mp3_path = os.path.join(mp3_directory, filename)
        convert_to_wav(mp3_path, output_directory)
