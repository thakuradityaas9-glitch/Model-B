# SaferoadAI

SaferoadAI is a real-time car accident detection system that utilizes a YOLO model trained on a custom dataset. The system is designed to work with a video feed from **CCTV or live stream**, detecting accidents in real-time and sending messages to the nearest hospital using the **OLA Maps API**.

## Features

- **Real-time accident detection** using YOLOv8.
- **Custom-trained model** on a car accident dataset.
- **Integration with CCTV or live video feeds** for practical applications.
- **Automatic alert system** using OLA Maps API to notify the nearest hospital.
- **Frame capture and image conversion** to URL for message attachments.
- **Potential for further enhancements**, such as traffic control automation.

## Installation

### Prerequisites

Ensure you have the following installed before proceeding:

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Streamlit (for deployment, if needed)
- OLA Maps API access

### Clone the Repository

```bash
git clone https://github.com/aayush010904/SaferoadAI.git
cd SaferoadAI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Datset

Dataset used for training : [Roboflow datset URL](https://universe.roboflow.com/accident-detection-model/accident-detection-model/dataset/2)

## Usage

### Running the Application

To start the accident detection system, run:

```bash
python app.py
```

### How It Works

- `app.py` imports functions from `SendMessage.py` and `NearestHospital.py` to send messages and fetch the nearest hospital.
- When an accident is detected, the frame is saved.
- The saved frame is converted into a URL using `Image2Url.py`.
- The image URL is sent along with an alert message to the nearest hospital.

### Training the YOLOv8 Model (If Needed)

```bash
python model_training.ipynb
```

## Flask Live Stream Demo (uses app.py logic)

The Flask app reuses the same YOLO model and detection logic defined in app.py for live MJPEG streaming.

Run locally:

```bash
pip install -r requirements.txt
python flask_app.py
```

Then open http://localhost:5000, upload or pick a sample video, click **Process Detection** to start the live processed stream. Accident counts appear after the stream completes. The “See live alerts” button is a placeholder—point it to your real alerts/messages URL.

## Deployment

### Quick local server (Waitress)

```bash
pip install waitress
waitress-serve --port=5000 flask_app:app
```

### Container (recommended for hosting)

Example Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "flask_app:app"]
```

Build and run:

```bash
docker build -t saferoad-ai .
docker run -p 5000:5000 saferoad-ai
```

Host this container on services like Render, Railway, Fly.io, AWS ECS/Fargate, or Azure Web App for Containers. Ensure ffmpeg/H.264 support is available in your base image if you need MP4 output.

## Project Structure

```
SaferoadAI/
├── app.py                # Main application script
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .env                  # Environment variables
├── best.pt               # Pre-trained YOLO model
├── best_model.pt         # Additional trained model
├── model_training.ipynb  # YOLO model training notebook
├── Image2Url.py          # Converts detected accident frames to image URLs
├── NearestHospital.py    # Fetches nearest hospital using OLA Maps API
├── SendMessage.py        # Sends alert messages with accident details
├── currentLocation.py    # Determines the user's current location
└── other_files/          # Additional scripts or resources
```

## Future Enhancements

- Improving model accuracy with more training data.
- Expanding API support for other mapping services.
- Implementing real-time traffic management integration.

## Contributions

Feel free to open an issue or submit a pull request if you’d like to contribute!

## License

This project is licensed under the MIT License.
