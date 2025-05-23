import numpy as np
import sounddevice as sd
import librosa
import pickle
import time
import os
import soundfile as sf
from sklearn.preprocessing import StandardScaler

# Constants
SAMPLE_RATE = 22050
DURATION = 3  # Record for 3 seconds
FIXED_LENGTH = SAMPLE_RATE * DURATION


# Function to extract features (same as in your training code)
def extract_features(data, sample_rate):
    # Standardize signal length
    if len(data) > FIXED_LENGTH:
        data = data[:FIXED_LENGTH]
    else:
        # Pad with zeros if shorter
        padding = np.zeros(FIXED_LENGTH - len(data))
        data = np.concatenate([data, padding])

    result = []

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result.append(zcr)

    # Spectral features
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result.append(chroma_stft)

    # MFCC (more detailed)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    result.append(mfcc)

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result.append(rms)

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result.append(mel)

    # Additional features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    result.append(spectral_centroid)

    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
    result.append(spectral_rolloff)

    return np.hstack(result)


# Function to record audio
def record_audio():
    print("Recording will start in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("Recording... Speak now!")

    # Record audio
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished

    print("Recording complete!")

    # Convert to the right format
    audio_data = recording.flatten()

    # Save the recording temporarily
    temp_file = "temp_recording.wav"
    sf.write(temp_file, audio_data, SAMPLE_RATE)
    print(f"Audio saved to {temp_file}")

    return audio_data, temp_file


# Function to load the trained model and scaler
def load_model_and_scaler():
    try:
        # Load MLP model
        mlp_model = pickle.load(open('mlp_model.pkl', 'rb'))
        print("MLP model loaded successfully!")

        # Load scaler
        scaler = pickle.load(open('scaler_mfcc.pkl', 'rb'))
        print("Feature scaler loaded successfully!")

        # Try to load emotion mapping if available
        try:
            emotions_map = pickle.load(open('emotion_mapping.pkl', 'rb'))
            print("Emotion mapping loaded successfully!")
        except FileNotFoundError:
            # Fallback mapping if file doesn't exist
            print("No emotion mapping file found, using default mapping.")
            emotions_map = {
                0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry',
                5: 'fear', 6: 'disgust', 7: 'surprise'
            }

        return mlp_model, scaler, emotions_map
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure to run your training code first to generate these files:")
        print("  - mlp_model.pkl: The trained MLP model")
        print("  - scaler_mfcc.pkl: The feature scaler")
        print("  - emotion_mapping.pkl: (Optional) The emotion label mapping")
        return None, None, None


# Main function
def predict_emotion():
    # Load the model and scaler
    mlp_model, scaler, emotions_map = load_model_and_scaler()
    if mlp_model is None or scaler is None:
        return

    while True:
        # Ask user if they want to record
        choice = input("\nDo you want to record and analyze your emotion? (y/n): ").lower()
        if choice != 'y':
            break

        # Record audio
        audio_data, temp_file = record_audio()

        # For better accuracy, reload the file with librosa
        y, sr = librosa.load(temp_file, sr=SAMPLE_RATE)

        # Extract features
        features = extract_features(y, sr)

        # Reshape features for the model
        features = features.reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict emotion
        prediction = mlp_model.predict(scaled_features)

        # Get prediction probabilities to show confidence
        prediction_proba = mlp_model.predict_proba(scaled_features)
        confidence = np.max(prediction_proba) * 100

        # Map prediction to emotion
        predicted_emotion = emotions_map.get(prediction[0], f"Unknown ({prediction[0]})")

        print(f"\n----- EMOTION PREDICTION -----")
        print(f"Your emotional state sounds like: {predicted_emotion.upper()}")
        print(f"Confidence: {confidence:.2f}%")

        # Show top 3 emotions if confidence is low
        if confidence < 70:
            print("\nAlternative possibilities:")
            # Get top 3 emotions
            top_indices = np.argsort(prediction_proba[0])[::-1][:3]
            for idx in top_indices:
                emotion = emotions_map.get(idx, f"Unknown ({idx})")
                prob = prediction_proba[0][idx] * 100
                print(f"  - {emotion.upper()}: {prob:.2f}%")

        print(f"--------------------------------")

        # Optionally play back the recording
        playback = input("Would you like to hear your recording? (y/n): ").lower()
        if playback == 'y':
            print("Playing back your recording...")
            data, sr = sf.read(temp_file)
            sd.play(data, sr)
            sd.wait()

    print("Thank you for using the emotion recognition system!")


if __name__ == "__main__":
    print("=" * 50)
    print("VOICE EMOTION RECOGNITION SYSTEM")
    print("=" * 50)
    print("This program will record your voice and predict your emotion.")
    print("Make sure you're in a quiet environment for better results.")
    print("=" * 50)

    predict_emotion()