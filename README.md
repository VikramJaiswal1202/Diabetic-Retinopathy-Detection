Table of contents

Project overview

Requirements

Folder / dataset layout (expected)

Create a virtual environment (venv)

Install dependencies (requirements.txt)

Common commands (train / eval / infer / streamlit)

Tips — GPU, data transforms, checkpoints

Troubleshooting

License & credits

Project overview

This repository contains a convolutional neural network (CNN) pipeline to detect/predict diabetic retinopathy from retinal images, plus a Streamlit web app to interactively run inference and view results. The codebase uses PyTorch and common data / plotting libraries.

Requirements

Python 3.9+ (3.10 recommended)

venv (or any virtualenv manager such as virtualenv or conda)

Optional: NVIDIA GPU + CUDA drivers if training on GPU (see Tips below)

The code imports the following libraries in the main notebooks / scripts (you provided):

    Python 3.9+ (3.10 recommended)

venv (or any virtualenv manager such as virtualenv or conda)

Optional: NVIDIA GPU + CUDA drivers if training on GPU (see Tips below)
