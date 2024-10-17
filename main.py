import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from preprocessor import AudioPreprocessor
from happytransformer import HappyTextToText, TTSettings

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

# Load grammar correction model (T5)
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
grammar_args = TTSettings(num_beams=5, min_length=1)

# Initialize the audio preprocessor
audio_preprocessor = AudioPreprocessor(target_sample_rate=16000)

# Function to transcribe audio
def transcribe_audio(file_path):
    # Load and preprocess the audio
    audio_input, sample_rate = audio_preprocessor.load_and_preprocess(file_path)

    # Process the audio input
    input_values = processor(
        audio_input,
        return_tensors="pt",
        padding="longest",
        sampling_rate=16000,  # Sampling rate
        feature_size=1         # Feature size (for mono audio)
    ).input_values

    # Get predictions
    with torch.no_grad():
        logits = model(input_values).logits

    # Get the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the IDs to text
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Function to perform grammar correction
def correct_grammar(transcription):
    result = happy_tt.generate_text(f"grammar: {transcription}", args=grammar_args)
    return result.text

# Example usage
if __name__ == "__main__":
    # Replace 'path_to_audio_file.wav' with your audio file path
    transcription = transcribe_audio("/Users/unico-095/Desktop/speech_to_text/python_app/xlsr-53-eng-wav2vec/Hutatma Gurunath Warde Marg.wav")  # Provide your audio file path here
    print("Transcription:", transcription)
    grammar = correct_grammar(transcription)
    print("Grammar Corrected Version:", grammar)
