import torch
import torchaudio
import os
import numpy as np
import sounddevice as sd
import wave
from speechbrain.inference import EncoderClassifier

# Load the Speaker Recognition Model
# classifier = EncoderClassifier.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb",
#     savedir="tmp_model"
# )
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="F:/rough use of project/tmp_model",  # Manually set path
    run_opts={"local_strategy": "copy"}  # Disable symlink
)
# Function to record audio
def record_audio(filename, duration=10, fs=16000):
    print("Recording... Please speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    
    # Save as WAV file
    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(fs)
    wavefile.writeframes(audio.tobytes())
    wavefile.close()
    print(f"Recording saved as {filename}")

# Function to extract voiceprint
def extract_voiceprint(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal)
    return embeddings.squeeze(0).squeeze(0).detach().numpy()

# Function to enroll a user
def enroll_user(user_name):
    user_filename = user_name.lower().replace(" ", "_") + ".wav"
    user_filepath = os.path.join("voiceprints", user_filename)

    record_audio(user_filepath)  # Record user's voice
    voiceprint = extract_voiceprint(user_filepath)

    npy_filename = user_filename.replace(".wav", ".npy")
    np.save(os.path.join("voiceprints", npy_filename), voiceprint)

    print(f"Voiceprint saved for {user_name}!")

# Create 'voiceprints' folder if not exists
os.makedirs("voiceprints", exist_ok=True)

# Enroll a user
user_name = input("Enter the user name: ")
enroll_user(user_name)
