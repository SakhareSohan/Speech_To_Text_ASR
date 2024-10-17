import soundfile as sf
import librosa

class AudioPreprocessor:
    def __init__(self, target_sample_rate=16000):
        self.target_sample_rate = target_sample_rate

    def normalize_audio(self, audio):
        """Normalizes the audio."""
        return audio / max(abs(audio)) if max(abs(audio)) > 0 else audio

    def load_and_preprocess(self, file_path):
        """Loads and preprocesses the audio file."""
        # Load audio file
        audio_input, sample_rate = sf.read(file_path)

        # Resample audio to the target sample rate if necessary
        if sample_rate != self.target_sample_rate:
            audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=self.target_sample_rate)
            sample_rate = self.target_sample_rate  # Update sample rate

        # Normalize the audio
        audio_input = self.normalize_audio(audio_input)

        return audio_input, sample_rate