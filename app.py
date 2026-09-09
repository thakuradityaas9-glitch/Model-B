import cv2
import torch
from ultralytics import YOLO
import os
import time
import numpy as np

# Load trained YOLO model
model = YOLO("best_model.pt")

# Define parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
FRAME_THRESHOLD = 5  # Number of consecutive frames needed to confirm an accident
NO_ACCIDENT_THRESHOLD = 30  # Frames without accident before resetting detection


def save_accident_frame(accident_frame, output_dir="accident_frames"):
    """Saves the accident frame and returns its file path."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    frame_filename = os.path.join(output_dir, f"accident_{int(time.time())}.jpg")
    cv2.imwrite(frame_filename, accident_frame)
    print(f"✅ Accident frame saved: {frame_filename}")
    return frame_filename  # Return the saved image path

def detect_accident(video_path, output_dir="accident_frames"):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    accident_frame_buffer = []
    detected = False  # Track if an accident has been confirmed
    no_accident_frames = 0  # Counter for frames without an accident

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        original_frame = frame.copy()  # Keep a copy for side-by-side display

        results = model(frame)  # Run YOLO detection
        accident_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])  # Get detected class
                conf = float(box.conf[0])  # Get confidence score

                if cls == 0 and conf > CONFIDENCE_THRESHOLD:  # Assuming class 0 is "accident"
                    accident_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Draw bounding box
                    cv2.putText(frame, f"Accident ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    break  # Stop after detecting one accident

        if accident_detected:
            accident_frame_buffer.append(frame)  # Store frame in buffer
            no_accident_frames = 0  # Reset counter

            # If accident is confirmed in `FRAME_THRESHOLD` consecutive frames, save & notify
            if len(accident_frame_buffer) >= FRAME_THRESHOLD and not detected:
                detected = True  # Mark accident as detected

                # Save the first frame of the accident **immediately**
                accident_frame = accident_frame_buffer[0]
                frame_filename = os.path.join(output_dir, f"accident_{int(time.time())}.jpg")
                cv2.imwrite(frame_filename, accident_frame)
                print(f"✅ Accident frame saved: {frame_filename} ")

                print(f"Accident confirmed locally: {frame_filename}")

        else:
            no_accident_frames += 1
            accident_frame_buffer = []  # Reset buffer

        # Reset detection after NO_ACCIDENT_THRESHOLD frames with no accident
        if no_accident_frames >= NO_ACCIDENT_THRESHOLD:
            detected = False  # Ready to detect the next accident

        # Resize frames for side-by-side display (Make sure both have the same height)
        frame_resized = cv2.resize(frame, (640, 360))
        original_resized = cv2.resize(original_frame, (640, 360))

        # Overlay text on both frames
        cv2.putText(original_resized, "Original Video", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame_resized, "Processed Video", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if detected:
            cv2.putText(frame_resized, "WASTED!!", (50, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        combined_frame = np.hstack((original_resized, frame_resized))  # Side-by-side

        # Display the frames in real-time
        cv2.imshow("Accident Detection | Left: Original | Right: Processed", combined_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # Close OpenCV windows properly

# Run detection
# Example usage
# video_path = "C:\\Users\\ASUS\\Desktop\\final_test.mp4"
if __name__ == "__main__":
    video_path="/Users/apple/Desktop/SafeRoadAI_model/SaferoadAI/sample_videos/video1.mp4"
    # video_path = "C:\\Users\\ASUS\\Desktop\\dash_cam.mp4"
    # video_path = "C:\\Users\\ASUS\\Desktop\\game_demo.mp4"
    detect_accident(video_path)

