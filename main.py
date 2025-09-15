import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from preprocessor import AudioPreprocessor
from happytransformer import HappyTextToText, TTSettings
import argparse
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(audio_path):
    """
    Main function to transcribe and correct an audio file.
    """
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found at: {audio_path}")
        return

    # --- Load Models (Consider moving this out if processing many files) ---
    logging.info("Loading speech-to-text model...")
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    
    logging.info("Loading grammar correction model...")
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    grammar_args = TTSettings(num_beams=5, min_length=1)

    # --- Preprocessing ---
    logging.info(f"Processing audio file: {audio_path}")
    audio_preprocessor = AudioPreprocessor(target_sample_rate=16000)
    try:
        audio_input, _ = audio_preprocessor.load_and_preprocess(audio_path)
    except Exception as e:
        logging.error(f"Failed to process audio file. Error: {e}")
        return

    # --- Transcription ---
    logging.info("Transcribing audio...")
    input_values = processor(
        audio_input,
        sampling_rate=16000,
        return_tensors="pt",
        padding="longest"
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    logging.info(f"Raw Transcription: {transcription}")

    # --- Grammar Correction ---
    logging.info("Correcting grammar...")
    # The model expects a prefix "grammar: "
    corrected_result = happy_tt.generate_text(f"grammar: {transcription}", args=grammar_args)
    
    print("\n--- RESULTS ---")
    print(f"Original Transcription:\n{transcription}")
    print(f"\nGrammar Corrected Version:\n{corrected_result.text}")
    print("---------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio and correct grammar.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to process.")
    
    args = parser.parse_args()
    main(args.audio_path)