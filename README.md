# Audio Emotion Detection

This project implements an audio-based emotion detection system using deep learning, specifically designed to work with the CREMA-D dataset. The system can recognize six different emotions: anger, disgust, fear, happiness, neutral, and sadness.

## Features

- Audio feature extraction using librosa
- MLP-based deep learning model for emotion classification
- Support for both training and inference
- Comprehensive evaluation metrics and visualizations
- Easy-to-use command-line interface

## Project Structure

```
.
├── README.md
├── requirements.txt
├── main.py
├── model.py
├── utils.py
├── data_processor.py
└── data/
    └── tabulatedVotes.csv
```

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
sudo apt-get install -y python3-soundfile python3-audioread
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && sudo apt-get install git-lfs

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd audio-emotion-detection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model on the CREMA-D dataset:

```bash
python main.py train --data_dir /path/to/data --model_save_path /path/to/save/model
```

### Prediction

To predict emotion from an audio file:

```bash
python main.py predict --model_path /path/to/model --audio_file /path/to/audio
```

## Feature Extraction

The system extracts the following audio features:

- Mel-frequency cepstral coefficients (MFCCs)
- Mel spectrogram
- Chroma features
- Spectral contrast

## Performance Metrics

The system generates the following evaluation metrics:

- Classification accuracy
- Confusion matrix
- Per-class precision, recall, and F1-score
- Training history plots

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- CREMA-D dataset creators and contributors
- librosa library for audio processing