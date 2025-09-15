# Audio Transcription and Grammar Correction Tool



This project is a command-line tool that transcribes spoken English from an audio file and then corrects the grammar of the transcribed text.

It uses the `jonatasgrosman/wav2vec2-large-xlsr-53-english` model from Hugging Face for speech-to-text and the `vennify/t5-base-grammar-correction` model for grammar correction.

## Features

-   **High-Accuracy Transcription**: Leverages the powerful Wav2Vec2 model.
-   **Automatic Grammar Correction**: Cleans up transcription errors using a T5 model.
-   **Audio Preprocessing**: Automatically resamples audio to the required 16kHz sample rate.

## Project Structure

```
.
├── main.py               # Main script to run the pipeline
├── preprocessor.py       # Class for loading and preprocessing audio
├── environment.yml       # Conda environment definition (Recommended)
├── requirements.txt      # Pip dependencies (Alternative)
└── README.md             # This file
```

## Setup & Installation

This project uses Conda for environment management to ensure PyTorch and its dependencies are handled correctly.

### Prerequisites

-   [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate the Conda environment:**
    The `environment.yml` file contains all the necessary dependencies. Create the environment using this file:
    ```bash
    conda env create -f environment.yml
    ```
    This command will create a new environment named `audio_project` and install all required packages.

3.  **Activate the new environment:**
    ```bash
    conda activate audio_project
    ```

## Usage

Run the `main.py` script from your terminal, passing the path to the audio file you want to process as an argument.

```bash
python main.py /path/to/your/audio.wav
```

### Example

```bash
python main.py ./samples/my_speech.wav
```

**Output:**

```
--- RESULTS ---
Original Transcription:
I HAS A APPLE AND TWO BANANA

Grammar Corrected Version:
I have an apple and two bananas.
---------------
```