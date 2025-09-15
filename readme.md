# Audio Transcription and Grammar Correction Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Hugging%20Face%20%7C%20PyTorch-orange.svg)
![Libraries](https://img.shields.io/badge/Libraries-Transformers%20%7C%20Librosa-green.svg)

This project is a powerful command-line tool that creates a two-stage pipeline to process spoken English from an audio file: first transcribing it to text, and then automatically correcting the grammar of the transcription.

## 📌 Project Motivation

Raw output from speech-to-text models is often filled with grammatical errors, disfluencies, and awkward phrasing, making it unsuitable for direct use. This tool solves that problem by creating an automated pipeline that not only provides a high-accuracy transcription but also refines it into clean, grammatically correct text ready for any application.



---

## ✨ Features

-   🗣️ **High-Accuracy Transcription**: Leverages the powerful `Wav2Vec2` model from Hugging Face for state-of-the-art Automatic Speech Recognition (ASR).
-   ✍️ **Automatic Grammar Correction**: Utilizes a sophisticated `T5 (Text-to-Text Transfer Transformer)` model specifically fine-tuned for grammar correction.
-   🎧 **Intelligent Audio Preprocessing**: Automatically loads, decodes, and resamples any input audio file to the required 16kHz sample rate for model compatibility.
-   ⚙️ **Efficient CLI Tool**: A simple and effective command-line interface for easy integration into scripts and workflows.

---

## 🛠️ How It Works: The Two-Stage Pipeline

The tool processes audio in two distinct stages:

### Stage 1: Speech-to-Text with Wav2Vec2
The input audio file is first loaded and preprocessed. It's then passed through the `jonatasgrosman/wav2vec2-large-xlsr-53-english` model, which analyzes the audio waveforms and converts the spoken words into a raw text transcription.

### Stage 2: Grammar Correction with T5
The raw, often imperfect, text from Stage 1 is then fed into the `vennify/t5-base-grammar-correction` model. This powerful transformer model analyzes the text and outputs a new, grammatically correct version.

---

## 💡 Key Skills & Concepts Demonstrated

-   **Natural Language Processing (NLP)**: Implementing a pipeline for text generation and correction.
-   **Audio Processing**: Loading, resampling, and preparing audio data for machine learning models using `librosa`.
-   **Deep Learning & Transformers**: Utilizing two different state-of-the-art transformer architectures (Wav2Vec2 for ASR and T5 for sequence-to-sequence correction).
-   **Model Integration**: Effectively using pre-trained models from the Hugging Face ecosystem.
-   **Environment Management**: Using `Conda` to manage complex dependencies, particularly for PyTorch.
-   **Command-Line Interface (CLI) Development**: Building a functional and user-friendly script for terminal-based execution.

---

## 📂 Project Structure

```
.
├── main.py               # Main script to run the pipeline
├── preprocessor.py       # Class for loading and preprocessing audio
├── environment.yml       # Conda environment definition (Recommended)
├── requirements.txt      # Pip dependencies (Alternative)
└── README.md             # This file
```

---

## 🚀 Setup & Installation

This project uses **Conda** for environment management to ensure PyTorch and its dependencies are handled correctly.

### Prerequisites
-   [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate the Conda environment:**
    The `environment.yml` file contains all necessary dependencies.
    ```bash
    # Create the environment from the file
    conda env create -f environment.yml

    # Activate the new environment
    conda activate audio_project
    ```

### Alternative: Using `pip`
A `requirements.txt` file is also provided. While Conda is recommended, you can try installing with pip in a virtual environment.
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the `main.py` script from your terminal, passing the path to your audio file (`.wav`, `.mp3`, etc.) as a command-line argument.

```bash
python main.py /path/to/your/audio.wav
```

### Example

```bash
python main.py ./samples/my_speech.wav
```

**Example Output:**
```
--- RESULTS ---
Original Transcription:
i has a apple and two banana

Grammar Corrected Version:
I have an apple and two bananas.
---------------
```

---