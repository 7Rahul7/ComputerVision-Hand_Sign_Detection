
# Hand Sign Detection

This project is a **real-time hand sign detection system** that can detect and classify different hand gestures from a webcam feed. It uses **computer vision** and **machine learning** to recognize hand shapes corresponding to specific signs.

---

## Features

* **Data Collection** – Capture custom hand sign images using your webcam.
* **Model Training** – Train a deep learning model to recognize different hand signs.
* **Real-time Prediction** – Detect and classify hand signs from live webcam video.

---

## Project Structure

| File / Folder  | Purpose                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `collect.py`   | Script for **collecting hand sign images** from your webcam. You can create your own dataset for different signs. |
| `train.py`     | Script for **training the hand sign detection model** using the collected dataset.                                |
| `predict.py`   | Script for **real-time prediction** of hand signs using a trained model.                                          |
| `models/`      | Stores the trained model files.                                                                                   |
| `data/`        | Stores collected hand sign images, organized into folders for each class.                                         |


---

## How It Works

| Step                        | Description                                                                                                                                                                   |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Data Collection**      | Run `collect.py` to capture images for each sign. The script uses your webcam to take pictures of your hand in different positions. Each sign is stored in a separate folder. |
| **2. Model Training**       | Run `train.py` to train a deep learning model (e.g., CNN) on your collected images. This model learns to recognize different hand shapes.                                     |
| **3. Real-time Prediction** | Run `predict.py` to start your webcam and detect hand signs in real-time using the trained model.                                                                             |

---

## Setup Instructions

This project uses **uv** for Python virtual environment management.

### 1. Install `uv`

If you don't already have **uv** installed, run:

```bash
pip install uv
```

### 2. Create and Activate Virtual Environment

```bash
uv venv
source .venv/bin/activate   # For Linux/Mac
.venv\Scripts\activate      # For Windows
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Run Scripts

**Collect data:**

```bash
python collect.py
```

**Train the model:**

```bash
python train.py
```

**Predict in real-time:**

```bash
python predict.py
```

---

## Requirements

The dependencies will be installed automatically via `uv sync`. Some common packages used in this project include:

| Library                 | Purpose                          |
| ----------------------- | -------------------------------- |
| `opencv-python`         | Webcam access & image processing |
| `tensorflow` or `torch` | Deep learning model training     |
| `mediapipe`             | Hand landmark detection          |
| `numpy`                 | Numerical computations           |

---

## Notes

* Ensure your **webcam** is working before running any scripts.
* The more **varied and clean** your dataset, the better the detection accuracy.
* You can modify `collect.py` to add more classes for new hand signs.
