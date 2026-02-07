
import os
import librosa
import numpy as np
import tensorflow as tf

# Hides background system messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LocalSER:
    def __init__(self, model_name):
        """Load the brain you trained in Kaggle."""
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"❌ Could not find {model_name} in this folder!")
        
        print(f"🔄 Loading model: {model_name}...")
        # compile=False makes it load faster since we aren't training anymore
        self.model = tf.keras.models.load_model(model_name, compile=False)
        
        # Standard RAVDESS emotion mapping
        self.emotions = {
            0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 
            4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
        }
        self.fixed_width = 150 

    def process_audio(self, path):
        """Convert your voice file into a Mel-Spectrogram 'image'."""
        y, sr = librosa.load(path, sr=22050)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        
        # Create the spectrogram
        mel = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=128)
        db_mel = librosa.power_to_db(mel, ref=np.max)
        
        # Adjust width to 150 (Padding/Cropping)
        if db_mel.shape[1] < self.fixed_width:
            pad = self.fixed_width - db_mel.shape[1]
            db_mel = np.pad(db_mel, ((0,0), (0,pad)), mode='constant', constant_values=-80)
        else:
            db_mel = db_mel[:, :self.fixed_width]
            
        # Scaling (0 to 1)
        db_mel = (db_mel - db_mel.min()) / (db_mel.max() - db_mel.min())
        
        # Reshape for CNN: (Batch, Height, Width, Channels)
        return db_mel.reshape(1, 128, 150, 1)

    def predict(self, audio_file):
        """Analyze the audio and print the result."""
        try:
            features = self.process_audio(audio_file)
            preds = self.model.predict(features, verbose=0)
            
            idx = np.argmax(preds)
            confidence = np.max(preds) * 100
            emotion = self.emotions[idx]
            
            print("\n" + "═"*35)
            print(f"🎤 AUDIO: {os.path.basename(audio_file)}")
            print(f"🧠 RESULT: {emotion.upper()}")
            print(f"📊 CONFIDENCE: {confidence:.2f}%")
            print("═"*35)
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Update this to the name of your local .wav file
    FILE_TO_TEST = "my_voice.wav" 
    
    # 2. Run the predictor
    ser = LocalSER("emotion_model_v1.h5")
    ser.predict(FILE_TO_TEST)