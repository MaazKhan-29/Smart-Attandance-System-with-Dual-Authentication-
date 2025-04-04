# import torch
# import torchaudio
# import os
# import numpy as np
# import sounddevice as sd
# import wave
# from speechbrain.inference import EncoderClassifier

# # Load the Speaker Recognition Model
# classifier = EncoderClassifier.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb",
#     savedir="tmp_model"
# )

# # Function to record audio
# def record_audio(filename, duration=3, fs=16000):
#     print("Recording... Please speak now!")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()
    
#     # Save as WAV file
#     wavefile = wave.open(filename, 'wb')
#     wavefile.setnchannels(1)
#     wavefile.setsampwidth(2)
#     wavefile.setframerate(fs)
#     wavefile.writeframes(audio.tobytes())
#     wavefile.close()
#     print(f"Recording saved as {filename}")

# # Function to extract voiceprint
# def extract_voiceprint(audio_path):
#     signal, fs = torchaudio.load(audio_path)
#     embeddings = classifier.encode_batch(signal)
#     return embeddings.squeeze(0).squeeze(0).detach().numpy()

# # Function to recognize speaker
# def recognize_speaker():
#     test_wav = "test_audio.wav"
#     record_audio(test_wav)  # Record test voice

#     test_voiceprint = extract_voiceprint(test_wav)

#     min_distance = float('inf')
#     recognized_user = "Unknown"

#     for file in os.listdir("voiceprints"):
#         if file.endswith(".npy"):
#             stored_voiceprint = np.load(os.path.join("voiceprints", file))
#             distance = np.linalg.norm(test_voiceprint - stored_voiceprint)

#             if distance < min_distance:
#                 min_distance = distance
#                 recognized_user = file.replace(".npy", "").replace("_", " ").title()

#     print(f"Recognized Speaker: {recognized_user} (Similarity Score: {min_distance:.4f})")

# # Run speaker recognition
# recognize_speaker()






# import torch
# import torchaudio
# import os
# import numpy as np
# import sounddevice as sd
# import wave
# from speechbrain.inference import EncoderClassifier

# # Load the Speaker Recognition Model
# classifier = EncoderClassifier.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb",
#     savedir="tmp_model"
# )

# # Function to record audio
# def record_audio(filename, duration=3, fs=16000):
#     print("Recording... Please speak now!")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()
    
#     # Save as WAV file
#     wavefile = wave.open(filename, 'wb')
#     wavefile.setnchannels(1)
#     wavefile.setsampwidth(2)
#     wavefile.setframerate(fs)
#     wavefile.writeframes(audio.tobytes())
#     wavefile.close()
#     print(f"Recording saved as {filename}")

# # Function to extract voiceprint
# def extract_voiceprint(audio_path):
#     signal, fs = torchaudio.load(audio_path)
#     embeddings = classifier.encode_batch(signal)
#     return embeddings.squeeze(0).squeeze(0).detach().numpy()

# # Function to recognize speaker
# def recognize_speaker(threshold=10.0):  # Set a higher threshold for rejecting unknown voices
#     test_wav = "test_audio.wav"
#     record_audio(test_wav)  # Record test voice

#     test_voiceprint = extract_voiceprint(test_wav)

#     min_distance = float('inf')
#     recognized_user = "Unknown"

#     for file in os.listdir("voiceprints"):
#         if file.endswith(".npy"):
#             stored_voiceprint = np.load(os.path.join("voiceprints", file))
#             distance = np.linalg.norm(test_voiceprint - stored_voiceprint)

#             print(f"Comparing with {file.replace('.npy', '').title()}: Distance = {distance:.4f}")

#             if distance < min_distance:
#                 min_distance = distance
#                 recognized_user = file.replace(".npy", "").replace("_", " ").title()

#     # Reject unknown voices if the distance is too high
#     if min_distance > threshold:
#         print(f"❌ Unknown Speaker! (Distance: {min_distance:.4f})")
#     else:
#         print(f"✅ Recognized Speaker: {recognized_user} (Distance: {min_distance:.4f})")

# # Run speaker recognition
# recognize_speaker()




import torch
import torchaudio
import os
import numpy as np
import sounddevice as sd
import wave
from scipy.spatial.distance import cosine
from speechbrain.inference import EncoderClassifier

# Load the Speaker Recognition Model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model"
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

# Function to extract voiceprint (embedding)
# Function to extract voiceprint (embedding)
def extract_voiceprint(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal)
    return embeddings.squeeze().detach().numpy()  # Ensure it's 1D


# Function to recognize speaker using Cosine Similarity
def recognize_speaker(threshold=0.35):  # Adjust the threshold dynamically
    test_wav = "test_audio.wav"
    record_audio(test_wav)  # Record test voice

    test_voiceprint = extract_voiceprint(test_wav)

    best_match = None
    best_similarity = -1  # Start with the lowest similarity score

    for file in os.listdir("voiceprints"):
        if file.endswith(".npy"):
            stored_voiceprint = np.load(os.path.join("voiceprints", file))
            similarity = 1 - cosine(test_voiceprint, stored_voiceprint)  # Cosine similarity

            print(f"Comparing with {file.replace('.npy', '').title()}: Similarity = {similarity:.4f}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = file.replace(".npy", "").replace("_", " ").title()

    # If the similarity is above threshold, accept the match
    if best_match and best_similarity >= threshold:
        print(f"✅ Recognized Speaker: {best_match} (Similarity: {best_similarity:.4f})")
    else:
        print(f"❌ Unknown Speaker! (Similarity: {best_similarity:.4f})")

# Run speaker recognition
recognize_speaker()
