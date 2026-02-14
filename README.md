# 🎥 LipNet - End-to-End Lip Reading Model

An implementation of LipNet — a deep learning model that performs sentence-level lip reading directly from video inputs.

## 🚀 Overview

LipNet is an end-to-end deep learning model that reads lips by analyzing video sequences of mouth movements and converting them into text.

This project uses:
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN / GRU / LSTM)
- CTC Loss for sequence prediction

## 🧠 Model Architecture

The model consists of:

1. Spatio-temporal CNN layers (for feature extraction)
2. Bidirectional GRU layers (for sequence learning)
3. CTC (Connectionist Temporal Classification) layer (for alignment-free training)

Pipeline:

Video Input → CNN → Bi-GRU → CTC → Predicted Sentence

## 📂 Project Structure

lipnet/
│
├── data/ # Dataset
├── models/ # Model architecture
├── training/ # Training scripts
├── utils/ # Helper functions
├── requirements.txt
└── main.py

This project uses the GRID Corpus dataset for training and evaluation.

Dataset contains:
- Video clips of speakers
- Corresponding text transcripts
