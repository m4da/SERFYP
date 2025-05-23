import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import librosa.feature
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from IPython.display import Audio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
from tslearn.metrics import dtw
from scipy.signal import resample
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Define dataset paths
Ravdess = "emotion_data/ravdess/audio_speech_actors_01-24/"
Crema = "emotion_data/crema/AudioWAV/"
Tess = "emotion_data/tess/TESS Toronto emotional speech set data/"
Savee = "emotion_data/savee/AudioData/"

# Set fixed length for audio signals (in samples)
# Common sampling rate is 22050 Hz, and for 3 seconds of audio:
FIXED_LENGTH = 22050 * 4


# Feature extraction functions
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


def extract_raw_audio(data, sample_rate):
    """Extract just the raw audio waveform, standardized to fixed length"""
    # More memory-efficient resampling
    if len(data) > FIXED_LENGTH:
        # Trim if too long
        data = data[:FIXED_LENGTH]
    elif len(data) < FIXED_LENGTH:
        # Pad with zeros if too short
        data = np.pad(data, (0, FIXED_LENGTH - len(data)))

    # Improved normalization for CNN processing
    # Normalize to range [-1, 1]
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))

    # Add extra preprocessing for CNN
    # Center the data around zero
    data = data - np.mean(data)

    # Apply standard scaling to help with convergence
    if np.std(data) > 0:
        data = data / np.std(data)

    return data


def augment_data(data, sample_rate):
    augmented_data = []
    # Original data
    augmented_data.append(data)

    # Noise injection
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    noise_data = data + noise_amp * np.random.normal(size=data.shape[0])
    augmented_data.append(noise_data)

    # Time stretching
    stretched_data = librosa.effects.time_stretch(data, rate=0.8)
    # Ensure fixed length after stretching
    if len(stretched_data) > FIXED_LENGTH:
        stretched_data = stretched_data[:FIXED_LENGTH]
    else:
        # Pad with zeros if shorter
        padding = np.zeros(FIXED_LENGTH - len(stretched_data))
        stretched_data = np.concatenate([stretched_data, padding])
    augmented_data.append(stretched_data)

    # Pitch shifting with better parameters
    pitch_shifted = librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=0.7)
    augmented_data.append(pitch_shifted)

    # Time shifting
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    shifted = np.roll(data, shift_range)
    augmented_data.append(shifted)

    return augmented_data


def process_audio_files(data_paths, feature_type='mfcc', augment=True):
    features, raw_audio, labels = [], [], []
    total_files = len(data_paths)

    for idx, (path, emotion) in enumerate(zip(data_paths['Path'], data_paths['Emotions']), 1):
        if idx % 100 == 0:
            print(f"Processing file {idx}/{total_files}")

        data, sample_rate = librosa.load(path, duration=3, offset=0.5)

        # Store raw audio (aligned to same length)
        aligned_audio = extract_raw_audio(data, sample_rate)

        if augment:
            for augmented in augment_data(data, sample_rate):
                if feature_type != 'raw':
                    features.append(extract_features(augmented, sample_rate))
                raw_audio.append(extract_raw_audio(augmented, sample_rate))
                labels.append(emotion)
        else:
            if feature_type != 'raw':
                features.append(extract_features(data, sample_rate))
            raw_audio.append(aligned_audio)
            labels.append(emotion)

    if feature_type == 'raw':
        return np.array(raw_audio, dtype=np.float32), np.array(labels)
    else:
        return np.array(features), np.array(labels)


# Helper function to preprocess data
def load_emotion_data(directory, file_processor):
    file_emotion = []
    file_path = []
    directory_list = os.listdir(directory)
    for item in directory_list:
        file_processor(item, file_emotion, file_path)
    return pd.DataFrame({'Emotions': file_emotion, 'Path': file_path})


# Processors for each dataset
def process_ravdess(item, file_emotion, file_path):
    actor_files = os.listdir(os.path.join(Ravdess, item))
    for file in actor_files:
        part = file.split('.')[0].split('-')
        file_emotion.append(int(part[2]))
        file_path.append(os.path.join(Ravdess, item, file))


def process_crema(file, file_emotion, file_path):
    file_path.append(os.path.join(Crema, file))
    part = file.split('_')
    if len(part) == 4:
        emotion_code = part[2]
        emotion_map = {
            'SAD': 'sad',
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral'
        }
        file_emotion.append(emotion_map.get(emotion_code, 'Unknown'))


def process_tess(item, file_emotion, file_path):
    dir_path = os.path.join(Tess, item)
    if os.path.isdir(dir_path):
        for subdir in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith('.wav'):
                        try:
                            emotion = file.split('_')[-1].split('.')[0]
                        except IndexError:
                            continue
                        file_emotion.append('surprise' if emotion == 'ps' else emotion)
                        file_path.append(os.path.join(subdir_path, file))


def process_savee(file, file_emotion, file_path):
    parts = file.split('_')
    if len(parts) > 1:
        emotion_code = parts[1]
        if emotion_code[-6:] == 'xx.wav':
            emotion_code = emotion_code[:-6]
        emotion_map = {
            'a': 'angry',
            'd': 'disgust',
            'f': 'fear',
            'h': 'happy',
            'n': 'neutral',
            'sa': 'sad'
        }
        file_emotion.append(emotion_map.get(emotion_code, 'surprise'))
        file_path.append(os.path.join(Savee, file))


def save_training_state(fold, epoch, model, optimizer, train_losses, val_losses,
                        train_accuracies, val_accuracies, best_val_acc, filename='training_state.pth'):
    torch.save({
        'fold': fold,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }, filename)


def load_training_state(model, optimizer, filename='training_state.pth'):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return (checkpoint['fold'], checkpoint['epoch'],
                checkpoint['train_losses'], checkpoint['val_losses'],
                checkpoint['train_accuracies'], checkpoint['val_accuracies'],
                checkpoint['best_val_acc'])
    return None


def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, title_suffix=""):
    plt.figure(figsize=(20, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training & Validation Loss {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Training & Validation Accuracy {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_results{title_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()


# Add confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, classes, title_suffix=""):
    """Plot confusion matrix with robust class handling"""
    # Make sure we only use classes that are actually in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    # Get class names if they're strings, otherwise use the class indices
    if isinstance(classes[0], str):
        class_names = [str(c) for c in unique_classes]
    else:
        class_names = [str(c) for c in unique_classes]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix {title_suffix}', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig(f'confusion_matrix{title_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()


# Additional visualization functions
def visualize_audio_sample(audio_file, emotion, save_path=None):
    """Creates comprehensive visualization of audio sample including waveform, spectrogram, and MFCCs"""
    plt.figure(figsize=(15, 10))

    # Load audio file
    y, sr = librosa.load(audio_file, duration=3)

    # Plot waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform - Emotion: {emotion}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')

    # Plot MFCCs
    plt.subplot(3, 1, 3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()


def visualize_feature_distributions(processed_data, feature_type='mfcc'):
    """Visualizes the distribution of features across different emotions"""
    X = processed_data[feature_type]['X_train']
    y = processed_data[feature_type]['y_train']

    # Get unique emotions
    unique_emotions = np.unique(y)

    if feature_type == 'mfcc':
        # Select some interesting features to visualize
        feature_indices = [0, 1, 2, 3]  # First few features
        feature_names = ['Zero Crossing Rate', 'Chroma', 'MFCC 1', 'MFCC 2']

        plt.figure(figsize=(15, 10))
        for i, (idx, name) in enumerate(zip(feature_indices, feature_names)):
            plt.subplot(2, 2, i + 1)
            for emotion in unique_emotions:
                emotion_data = X[y == emotion, idx]
                sns.kdeplot(emotion_data, label=emotion)
            plt.title(f'Distribution of {name}')
            plt.xlabel('Value')
            plt.ylabel('Density')

        plt.tight_layout()
        plt.legend(title='Emotion')
        plt.savefig(f'feature_distributions_{feature_type}.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_emotion_correlation_heatmap(processed_data, feature_type='mfcc'):
    """Creates a correlation heatmap between different features"""
    X = processed_data[feature_type]['X_train']

    # Skip for raw audio data due to memory constraints
    if feature_type == 'raw' or X.shape[1] > 1000:
        print(f"Skipping correlation heatmap for {feature_type} features (too large)")
        return None

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlation Heatmap - {feature_type.upper()}')
    plt.tight_layout()
    plt.savefig(f'correlation_heatmap_{feature_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return corr_matrix


def visualize_t_sne(processed_data, feature_type='mfcc', perplexity=30, n_iter=1000):
    """Creates t-SNE visualization of the feature space colored by emotions"""

    X = processed_data[feature_type]['X_train']
    y = processed_data[feature_type]['y_train']

    # If data is too large, sample it
    if len(X) > 5000:
        indices = np.random.choice(len(X), 5000, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y

    # Apply t-SNE
    print(f"Applying t-SNE to {feature_type} features...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)

    # Plot the results
    plt.figure(figsize=(12, 10))
    unique_emotions = np.unique(y_sample)
    for emotion in unique_emotions:
        indices = y_sample == emotion
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=emotion, alpha=0.7)

    plt.title(f't-SNE Visualization of {feature_type.upper()} Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Emotion')
    plt.tight_layout()
    plt.savefig(f'tsne_{feature_type}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves_comparison(feature_types, results_dict):
    """Plots comparative learning curves for different models/feature types"""
    plt.figure(figsize=(15, 10))

    # Subplot for accuracy
    plt.subplot(2, 1, 1)
    for feature_type in feature_types:
        if feature_type in results_dict:
            plt.plot(
                range(1, len(results_dict[feature_type]['train_accuracies']) + 1),
                results_dict[feature_type]['train_accuracies'],
                'o-',
                label=f"{feature_type} Train"
            )
            plt.plot(
                range(1, len(results_dict[feature_type]['val_accuracies']) + 1),
                results_dict[feature_type]['val_accuracies'],
                'o-',
                label=f"{feature_type} Validation"
            )

    plt.title('Training and Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Subplot for loss
    plt.subplot(2, 1, 2)
    for feature_type in feature_types:
        if feature_type in results_dict:
            plt.plot(
                range(1, len(results_dict[feature_type]['train_losses']) + 1),
                results_dict[feature_type]['train_losses'],
                'o-',
                label=f"{feature_type} Train"
            )
            plt.plot(
                range(1, len(results_dict[feature_type]['val_losses']) + 1),
                results_dict[feature_type]['val_losses'],
                'o-',
                label=f"{feature_type} Validation"
            )

    plt.title('Training and Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_model_performance_comparison(all_results):
    """Creates a bar chart comparing performance of different models"""
    models = list(all_results.keys())
    accuracies = list(all_results.values())

    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom')

    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_feature_importance_heatmap(rf_model, feature_names=None):
    """Creates a heatmap of feature importance from a Random Forest model"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Take top 20 features
    top_n = 20
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]

    # Get feature names if available, otherwise use indices
    if feature_names is not None and len(feature_names) == len(importances):
        top_features = [feature_names[i] for i in top_indices]
    else:
        top_features = [f"Feature {i}" for i in top_indices]

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        top_importances.reshape(1, -1),
        annot=True,
        fmt='.3f',
        cmap='viridis',
        xticklabels=top_features,
        yticklabels=['Importance']
    )
    plt.title('Feature Importance Heatmap')
    plt.tight_layout()
    plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_with_percentages(y_true, y_pred, classes, title_suffix=""):
    """Plot confusion matrix with percentage of correct predictions"""
    # Make sure we only use classes that are actually in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))

    # Get class names
    if isinstance(classes[0], str):
        class_names = [str(c) for c in unique_classes]
    else:
        class_names = [str(c) for c in unique_classes]

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot
    plt.figure(figsize=(12, 10))

    # Plot the raw counts
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Counts) {title_suffix}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Plot the percentages
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Percentages) {title_suffix}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.tight_layout()
    plt.savefig(f'confusion_matrix_detailed{title_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()


def add_visualizations(processed_data, feature_types, models_results):
    """
    Generate all visualizations for the portfolio

    Parameters:
    -----------
    processed_data : dict
        Dictionary containing processed features
    feature_types : list
        List of feature types used ('mfcc', 'raw')
    models_results : dict
        Dictionary containing model results
    """
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS FOR PORTFOLIO")
    print("=" * 50)

    # Generate model performance comparison
    print("Generating model performance visualization...")
    visualize_model_performance_comparison(models_results)

    # Generate feature visualizations for each feature type
    for feature_type in feature_types:
        if feature_type in processed_data:
            print(f"Generating visualizations for {feature_type} features...")

            # Feature distributions
            visualize_feature_distributions(processed_data, feature_type)

            # Feature correlation heatmap (only for MFCC, skip for raw)
            if feature_type != 'raw':
                create_emotion_correlation_heatmap(processed_data, feature_type)
            else:
                print(f"Skipping correlation heatmap for {feature_type} features (too memory intensive)")

            # t-SNE visualization
            try:
                visualize_t_sne(processed_data, feature_type)
            except Exception as e:
                print(f"Error generating t-SNE visualization: {e}")

    # Visualize sample audio files for each emotion
    print("Generating audio sample visualizations...")
    sample_emotions = {}

    # Find one sample file for each emotion
    for name, df in [('RAVDESS', Ravdess_df), ('CREMA', Crema_df),
                     ('TESS', Tess_df), ('SAVEE', Savee_df)]:
        for emotion in np.unique(df['Emotions']):
            if emotion not in sample_emotions and len(df[df['Emotions'] == emotion]) > 0:
                sample_file = df[df['Emotions'] == emotion]['Path'].iloc[0]
                sample_emotions[emotion] = sample_file

    # Generate visualizations for each emotion
    for emotion, file_path in sample_emotions.items():
        try:
            print(f"Visualizing audio sample for emotion: {emotion}")
            visualize_audio_sample(file_path, emotion, save_path=f'audio_sample_{emotion}.png')
        except Exception as e:
            print(f"Error visualizing audio for {emotion}: {e}")

    print("\nAll visualizations generated successfully!")
    print("Visualization files saved to disk for your portfolio.")


def save_model_with_metadata(model, optimizer, epoch, accuracy, loss, filename, feature_type='mfcc'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': loss,
        'feature_params': {
            'feature_type': feature_type,
            'augmentation_types': ['noise', 'stretch', 'pitch', 'shift']
        }
    }, filename)


# CNN model for traditional audio features
class EmotionCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Remove the Softmax from the last layer
        self.features = nn.Sequential(
            # First Conv Block
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),

            # Second Conv Block
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),

            # Third Conv Block
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.3),
        )

        # Calculate feature size properly using a dummy input tensor
        # Reshape to match the actual input format
        with torch.no_grad():
            dummy_input = torch.ones(1, 1, input_shape)  # Changed input_dim to input_shape
            features_output = self.features(dummy_input)
            self.flattened_size = features_output.view(1, -1).size(1)
            print(f"Calculated flattened size: {self.flattened_size}")

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
            # Remove the Softmax here!
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)  # Flatten all dimensions except batch
        x = self.classifier(x)
        return x


# CNN model that learns directly from raw audio
class RawAudioCNN(nn.Module):
    def __init__(self, input_length, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 128, kernel_size=80, stride=4, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            # Layer 2
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            # Layer 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            # Layer 4
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            # Layer 5
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Dropout(0.3),
        )

        # Calculate the output size
        with torch.no_grad():
            x = torch.randn(1, 1, input_length)
            x = self.features(x)
            flattened_size = x.view(1, -1).size(1)
            print(f"Calculated flattened size for raw audio: {flattened_size}")

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
            # Removed Softmax here to work better with CrossEntropyLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)  # Flatten all dimensions except batch
        x = self.classifier(x)
        return x


# Function to perform dynamic time warping for signal alignment
def align_signals_with_dtw(signal1, signal2):
    alignment = dtw(signal1.reshape(-1, 1), signal2.reshape(-1, 1))
    return alignment


def evaluate_traditional_ml(X_train, X_test, y_train, y_test, emotions_list, suffix=""):
    """Evaluate traditional ML models like KNN, Random Forest, and MLP with proper emotion class handling"""

    results = {}

    # Find which emotions are actually in the data
    present_classes = np.unique(np.concatenate([y_train, y_test]))
    present_indices = [i for i, label in enumerate(present_classes)]

    # Get the emotion names that are actually in the data
    emotions_present = [emotions_list[i] for i in present_indices]

    print(f"Note: Found {len(present_classes)} classes in the data out of {len(emotions_list)} possible classes")
    print(f"Classes present in data: {present_classes}")

    # 1. K-Nearest Neighbors
    print("\nTraining KNN model...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_accuracy = (y_pred == y_test).mean() * 100
    results['KNN'] = knn_accuracy

    print(f"KNN Accuracy: {knn_accuracy:.2f}%")
    try:
        # Use only the emotions present in the data for classification report
        print(classification_report(y_test, y_pred, labels=np.unique(y_test)))
    except Exception as e:
        print(f"Error generating classification report: {e}")
    plot_confusion_matrix(y_test, y_pred, np.unique(y_test), f"_KNN{suffix}")

    # 2. Random Forest
    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_accuracy = (y_pred == y_test).mean() * 100
    results['Random Forest'] = rf_accuracy

    print(f"Random Forest Accuracy: {rf_accuracy:.2f}%")
    try:
        print(classification_report(y_test, y_pred, labels=np.unique(y_test)))
    except Exception as e:
        print(f"Error generating classification report: {e}")
    plot_confusion_matrix(y_test, y_pred, np.unique(y_test), f"_RF{suffix}")

    # 3. MLP Classifier
    print("\nTraining MLP model...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300,
                        activation='relu', solver='adam', random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    mlp_accuracy = (y_pred == y_test).mean() * 100
    results['MLP'] = mlp_accuracy

    print(f"MLP Accuracy: {mlp_accuracy:.2f}%")
    try:
        print(classification_report(y_test, y_pred, labels=np.unique(y_test)))
    except Exception as e:
        print(f"Error generating classification report: {e}")
    plot_confusion_matrix(y_test, y_pred, np.unique(y_test), f"_MLP{suffix}")

    # NEW CODE: Save the MLP model specifically for the voice recognition app
    try:
        print("\nSaving MLP model for inference...")
        with open('mlp_model.pkl', 'wb') as f:
            pickle.dump(mlp, f)
        print("MLP model saved as 'mlp_model.pkl'")
    except Exception as e:
        print(f"Error saving MLP model: {e}")

    return results, knn, rf, mlp




print("Loading datasets...")

# Load datasets
Ravdess_df = load_emotion_data(Ravdess, process_ravdess)
Crema_df = load_emotion_data(Crema, process_crema)
Tess_df = load_emotion_data(Tess, process_tess)
Savee_df = load_emotion_data(Savee, process_savee)

Ravdess_df['Emotions'] = Ravdess_df['Emotions'].replace({
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry',
    6: 'fear', 7: 'disgust', 8: 'surprise'
})

# Get the size of each dataset
print(f"RAVDESS dataset size: {len(Ravdess_df)}")
print(f"CREMA dataset size: {len(Crema_df)}")
print(f"TESS dataset size: {len(Tess_df)}")
print(f"SAVEE dataset size: {len(Savee_df)}")

# Determine the two largest and one smallest datasets
dataset_sizes = {
    'RAVDESS': len(Ravdess_df),
    'CREMA': len(Crema_df),
    'TESS': len(Tess_df),
    'SAVEE': len(Savee_df)
}

# Combine all datasets
combined_df = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis=0)
print(f"Combined dataset size: {len(combined_df)}")

# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create an 85:15 train-test split
train_size = int(len(combined_df) * 0.85)
training_data = combined_df.iloc[:train_size]
testing_df = combined_df.iloc[train_size:]

print(f"Training data size: {len(training_data)} (85%)")
print(f"Testing data size: {len(testing_df)} (15%)")

# Save the split data for reference
training_data.to_csv("training_data.csv", index=False)
testing_df.to_csv("testing_data.csv", index=False)

# Save the updated split data for reference
training_data.to_csv("training_data.csv", index=False)
testing_df.to_csv("testing_data.csv", index=False)

# Process audio files for different feature types
feature_types = ['mfcc', 'raw']
processed_data = {}

# Replace your current feature loading section with this fixed version
for feature_type in feature_types:
    print(f"\nProcessing {feature_type} features...")

    # Flag to track if we need to process features
    need_processing = True

    try:
        # Try to load preprocessed features from cache
        print(f"Attempting to load preprocessed {feature_type} features...")

        if feature_type == 'raw':
            # For raw audio, try to load chunked files first
            metadata_file = f'processed_{feature_type}_X_train_metadata.pkl'
            if os.path.exists(metadata_file):
                try:
                    print(f"Found chunked data files. Loading chunks...")
                    # Load metadata
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)

                    # Initialize empty array for X_train
                    total_samples = metadata['total_samples']
                    feature_dim = metadata['feature_dim']
                    num_chunks = metadata['num_chunks']
                    chunk_size = metadata['chunk_size']

                    # Pre-allocate the array
                    X_train = np.zeros((total_samples, feature_dim), dtype=np.float32)

                    # Load each chunk
                    for i in range(num_chunks):
                        chunk_file = f'processed_{feature_type}_X_train_chunk_{i}.pkl'
                        if os.path.exists(chunk_file):
                            print(f"Loading chunk {i + 1}/{num_chunks}...")
                            with open(chunk_file, 'rb') as f:
                                chunk_data = pickle.load(f)

                            # Calculate indices
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, total_samples)

                            # Store chunk in the pre-allocated array
                            X_train[start_idx:end_idx] = chunk_data
                        else:
                            raise FileNotFoundError(f"Chunk file {chunk_file} not found")

                    # Load the other files
                    with open(f'processed_{feature_type}_y_train.pkl', 'rb') as f:
                        y_train = pickle.load(f)
                    with open(f'processed_{feature_type}_X_test.pkl', 'rb') as f:
                        X_test = pickle.load(f)
                    with open(f'processed_{feature_type}_y_test.pkl', 'rb') as f:
                        y_test = pickle.load(f)

                    processed_data[feature_type] = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test
                    }
                    print(f"Successfully loaded all chunked {feature_type} features!")
                    need_processing = False
                except Exception as e:
                    print(f"Error loading chunked files: {e}")
                    need_processing = True
            else:
                # If no metadata file, try loading single files
                try:
                    X_train = pickle.load(open(f'processed_{feature_type}_X_train.pkl', 'rb'))
                    y_train = pickle.load(open(f'processed_{feature_type}_y_train.pkl', 'rb'))
                    X_test = pickle.load(open(f'processed_{feature_type}_X_test.pkl', 'rb'))
                    y_test = pickle.load(open(f'processed_{feature_type}_y_test.pkl', 'rb'))
                    processed_data[feature_type] = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test
                    }
                    print(f"Successfully loaded preprocessed {feature_type} features from split files!")
                    need_processing = False
                except (FileNotFoundError, EOFError) as e:
                    print(f"Could not load split files: {e}")
                    need_processing = True
        else:
            # For MFCC features, use the original approach
            try:
                with open(f'processed_{feature_type}_features.pkl', 'rb') as f:
                    processed_data[feature_type] = pickle.load(f)
                print(f"Successfully loaded preprocessed {feature_type} features!")
                need_processing = False
            except (FileNotFoundError, EOFError) as e:
                print(f"Could not load combined file: {e}")
                need_processing = True
    except Exception as e:
        print(f"Unexpected error loading features: {e}")
        need_processing = True

    # Process features if needed
    if need_processing:
        print(f"Processing {feature_type} features for training data...")
        X_train, y_train = process_audio_files(training_data, feature_type=feature_type)

        print(f"Processing {feature_type} features for testing data...")
        X_test, y_test = process_audio_files(testing_df, feature_type=feature_type, augment=False)

        processed_data[feature_type] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }

        # Save processed features - with different approaches based on feature type
        if feature_type == 'raw':
            # For raw audio, save as separate files to avoid memory issues
            print(f"Saving {feature_type} features as separate files...")
            try:
                # First save in smaller chunks for X_train which is the largest
                print(f"Saving X_train (shape: {X_train.shape})...")

                # Save in chunks if the data is large
                if X_train.shape[0] > 5000:  # If more than 5000 samples
                    chunk_size = 5000
                    num_chunks = (X_train.shape[0] + chunk_size - 1) // chunk_size  # Ceiling division

                    print(f"X_train is large, saving in {num_chunks} chunks of {chunk_size} samples each")

                    for i in range(num_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, X_train.shape[0])

                        print(f"Saving chunk {i + 1}/{num_chunks} (samples {start_idx} to {end_idx})...")
                        with open(f'processed_{feature_type}_X_train_chunk_{i}.pkl', 'wb') as f:
                            pickle.dump(X_train[start_idx:end_idx], f)

                    # Save a metadata file with information about the chunks
                    with open(f'processed_{feature_type}_X_train_metadata.pkl', 'wb') as f:
                        pickle.dump({
                            'total_samples': X_train.shape[0],
                            'feature_dim': X_train.shape[1],
                            'num_chunks': num_chunks,
                            'chunk_size': chunk_size
                        }, f)
                else:
                    # Save as a single file if small enough
                    with open(f'processed_{feature_type}_X_train.pkl', 'wb') as f:
                        pickle.dump(X_train, f)

                # Save the rest of the data
                print("Saving y_train...")
                with open(f'processed_{feature_type}_y_train.pkl', 'wb') as f:
                    pickle.dump(y_train, f)

                print("Saving X_test...")
                with open(f'processed_{feature_type}_X_test.pkl', 'wb') as f:
                    pickle.dump(X_test, f)

                print("Saving y_test...")
                with open(f'processed_{feature_type}_y_test.pkl', 'wb') as f:
                    pickle.dump(y_test, f)

                print(f"Successfully saved all {feature_type} features!")
            except Exception as e:
                import traceback

                print(f"Error saving {feature_type} features: {e}")
                print("Detailed error:")
                traceback.print_exc()

# Define emotion classes consistently
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# Check for empty test set
for feature_type in feature_types:
    if feature_type in processed_data:
        X_test = processed_data[feature_type]['X_test']
        if X_test.shape[0] == 0:  # If test set is empty
            print(f"Warning: Test set for {feature_type} is empty. Creating test set from training data.")

            # Get training data
            X_train = processed_data[feature_type]['X_train']
            y_train = processed_data[feature_type]['y_train']

            # If we have enough training samples
            if len(X_train) >= 5:
                try:
                    # Count samples per class to ensure we have enough for stratification
                    unique_classes, class_counts = np.unique(y_train, return_counts=True)
                    num_classes = len(unique_classes)

                    print(
                        f"Training data has {num_classes} classes with counts: {dict(zip(unique_classes, class_counts))}")

                    # If we have at least 2 samples per class for all classes, use stratified split
                    if all(count >= 2 for count in class_counts):
                        # Calculate a reasonable test size
                        test_size = max(0.2, num_classes / len(X_train) * 2)  # At least 20% or 2 samples per class
                        print(f"Using stratified split with test_size={test_size:.2f}")

                        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
                            X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
                        )
                    else:
                        # If we don't have enough samples per class, use a simple random split
                        print("Not enough samples per class for stratification, using random split")
                        test_size = 0.2  # Use 20% of data for testing
                        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
                            X_train, y_train, test_size=test_size, random_state=42, stratify=None
                        )

                    # Update the processed data
                    processed_data[feature_type]['X_train'] = X_train_new
                    processed_data[feature_type]['y_train'] = y_train_new
                    processed_data[feature_type]['X_test'] = X_test_new
                    processed_data[feature_type]['y_test'] = y_test_new

                    print(f"Created test set from training data:")
                    print(f"New training data shape: {X_train_new.shape}")
                    print(f"New testing data shape: {X_test_new.shape}")

                except Exception as e:
                    print(f"Error creating test set: {str(e)}")
                    print("Using a simple random split instead")

                    # Fallback to simple random split with a fixed number of test samples
                    test_samples = min(int(len(X_train) * 0.2), 100)  # At most 100 samples or 20%
                    indices = np.random.permutation(len(X_train))
                    test_idx, train_idx = indices[:test_samples], indices[test_samples:]

                    X_test_new = X_train[test_idx]
                    y_test_new = y_train[test_idx]
                    X_train_new = X_train[train_idx]
                    y_train_new = y_train[train_idx]

                    # Update the processed data
                    processed_data[feature_type]['X_train'] = X_train_new
                    processed_data[feature_type]['y_train'] = y_train_new
                    processed_data[feature_type]['X_test'] = X_test_new
                    processed_data[feature_type]['y_test'] = y_test_new

                    print(f"Created test set using random split:")
                    print(f"New training data shape: {X_train_new.shape}")
                    print(f"New testing data shape: {X_test_new.shape}")
            else:
                print(f"Error: Not enough training samples to create a test set. Skip {feature_type}.")
                # Remove this feature type from the list to avoid further processing
                feature_types = [ft for ft in feature_types if ft != feature_type]

# Initialize dictionaries to store results
all_model_results = {}
learning_curves_data = {}
ml_models = {}

# Train and evaluate models
for feature_type in feature_types:
    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODELS WITH {feature_type.upper()} FEATURES")
    print(f"{'=' * 50}")

    X_train = processed_data[feature_type]['X_train']
    y_train = processed_data[feature_type]['y_train']
    X_test = processed_data[feature_type]['X_test']
    y_test = processed_data[feature_type]['y_test']

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Find all unique categories across both train and test sets
    all_categories = np.unique(np.concatenate([y_train, y_test]))
    print(f"All emotion categories: {all_categories}")

    # Encode labels with all possible categories (handle_unknown='ignore')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[all_categories])
    encoder.fit(all_categories.reshape(-1, 1))  # Fit on all possible categories

    # Transform the data
    y_train_encoded = encoder.transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

    # Get emotion mapping
    encoder_classes = encoder.categories_[0]
    print("Emotion mapping:", dict(enumerate(encoder_classes)))

    # NEW CODE: Save the emotion mapping for inference
    try:
        # Create a mapping from index to emotion name
        emotion_mapping = {i: emotion for i, emotion in enumerate(encoder_classes)}
        with open('emotion_mapping.pkl', 'wb') as f:
            pickle.dump(emotion_mapping, f)
        print("Emotion mapping saved as 'emotion_mapping.pkl' for inference")
    except Exception as e:
        print(f"Error saving emotion mapping: {e}")

    # Convert string labels to categorical indices for traditional ML
    y_train_indices = np.array([np.where(encoder_classes == label)[0][0] for label in y_train])
    y_test_indices = np.array([np.where(encoder_classes == label)[0][0] for label in y_test])

    # Scale features for MFCC and other extracted features (not for raw audio)
    if feature_type != 'raw':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the scaler
        with open(f'scaler_{feature_type}.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    if feature_type == 'mfcc':
        try:
            with open('scaler_mfcc.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print("Feature scaler saved as 'scaler_mfcc.pkl' for inference")
        except Exception as e:
            print(f"Error saving scaler for inference: {e}")

    # 1. Evaluate traditional ML models (only for MFCC features)
    if feature_type != 'raw':
        print("\nEvaluating traditional ML models...")
        ml_results, knn, rf, mlp = evaluate_traditional_ml(
            X_train, X_test, y_train_indices, y_test_indices, emotions, f"_{feature_type}"
        )
        ml_models[feature_type] = {'knn': knn, 'rf': rf, 'mlp': mlp}

        # Store ML results
        for model_name, accuracy in ml_results.items():
            all_model_results[f"{model_name} ({feature_type.upper()})"] = accuracy

    # 2. Train and evaluate deep learning models
    print("\nTraining deep learning model...")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train_encoded)
    y_test_tensor = torch.FloatTensor(y_test_encoded)

    if feature_type == 'raw':
        # For raw audio, the model will treat it as a 1D signal
        model = RawAudioCNN(X_train.shape[1], len(encoder.categories_[0])).to(device)
        # Reshape tensors to have a channel dimension
        X_train_tensor = X_train_tensor.unsqueeze(1)  # Add channel dimension
        X_test_tensor = X_test_tensor.unsqueeze(1)  # Add channel dimension
    else:
        # Add the channel dimension for Conv1D
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_test_tensor = X_test_tensor.unsqueeze(1)
        model = EmotionCNN(X_train_tensor.shape[2], len(encoder.categories_[0])).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)  # Lower learning rate
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6, verbose=True
    )

    # Lists for tracking metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0
    best_model_state = None

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y.argmax(1)).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y.argmax(1)).sum().item()

        val_loss = val_loss / len(test_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            save_model_with_metadata(
                model, optimizer, epoch, val_acc, val_loss,
                f'best_model_{feature_type}.pth',
                feature_type
            )

    # Save learning curves data for comparison
    learning_curves_data[feature_type] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

    # Load best model and evaluate
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    # Plot training history
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, f"_{feature_type}")

    # Evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_predictions = []
    test_true = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y.argmax(1)).sum().item()

            test_predictions.extend(predicted.cpu().numpy())
            test_true.extend(batch_y.argmax(1).cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    # Save CNN results
    all_model_results[f"CNN ({feature_type.upper()})"] = test_acc

    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Plot confusion matrix
    plot_confusion_matrix(test_true, test_predictions, emotions, f"_CNN_{feature_type}")

    # Also plot detailed confusion matrix with percentages
    plot_confusion_matrix_with_percentages(test_true, test_predictions, emotions, f"_CNN_{feature_type}")

    # Print classification report
    print("\nClassification Report:")
    present_classes = np.unique(test_true)
    target_names = [emotions[i] for i in present_classes]
    print(classification_report(test_true, test_predictions, target_names=target_names))

# Compare results of different approaches
print("\n" + "=" * 50)
print("FINAL RESULTS COMPARISON")
print("=" * 50)

# Add visualization for feature importance if using Random Forest
# This is applicable only for the MFCC features
if 'mfcc' in processed_data and 'rf' in ml_models.get('mfcc', {}):
    rf = ml_models['mfcc']['rf']

    # Re-train a Random Forest for feature importance
    X_train = processed_data['mfcc']['X_train']
    y_train = processed_data['mfcc']['y_train']

    # Convert string labels to indices
    encoder_classes = encoder.categories_[0]
    y_train_indices = np.array([np.where(encoder_classes == label)[0][0] for label in y_train])

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Train Random Forest if not already trained
    if rf is None:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train_indices)

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances', size=16)
    plt.bar(range(min(20, X_train.shape[1])), importances[indices[:20]], align='center')
    plt.xticks(range(min(20, X_train.shape[1])), indices[:20])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Also create the heatmap version
    create_feature_importance_heatmap(rf)

    print("\nFeature importance visualizations saved.")

    # Generate comprehensive visualizations for portfolio
    print("\nGenerating comprehensive visualizations for portfolio...")

    # Generate all visualizations
    add_visualizations(processed_data, feature_types, all_model_results)

    # Create combined learning curves comparison
    plot_learning_curves_comparison(feature_types, learning_curves_data)

    print("\nAll results and visualizations have been saved to disk.")
    print("This includes:")
    print("- Trained models for each feature type")
    print("- Confusion matrices for each model type (counts and percentages)")
    print("- Training history plots and combined learning curves")
    print("- Feature importance visualizations")
    print("- Audio sample visualizations for each emotion")
    print("- t-SNE visualizations of feature spaces")
    print("- Feature distributions and correlations")
    print("- Model performance comparison")

else:
# For MFCC, save as a single file
    try:
        print(f"Saving MFCC features (X_train shape: {X_train.shape}, X_test shape: {X_test.shape})...")
        with open(f'processed_{feature_type}_features.pkl', 'wb') as f:
            pickle.dump(processed_data[feature_type], f)
        print(f"Saved processed {feature_type} features to cache!")
    except Exception as e:
        import traceback

        print(f"Error saving {feature_type} features: {e}")
        print("Detailed error:")
        traceback.print_exc()