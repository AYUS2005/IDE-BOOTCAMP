import cv2
import numpy as np
import pygame
import time

# Initialize Pygame for sound
pygame.mixer.init()

# Load the alarm sound (Make sure 'alarm.mp3' is in the same directory)
alarm_sound = "alarm.mp3"

# Load the video file
video_path = "crowd.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened correctly
if not cap.isOpened():
    print("Error: Video file not found or couldn't be opened.")
    exit()

# Set cooldown time for alert (in seconds)
alert_cooldown = 7  # Updated to 7 seconds
last_alert_time = 0  # Track last alert time

# Loop through the video frames
while True:
    ret, frame = cap.read()

    # If the video ends, restart from the beginning
    if not ret:
        print("Reaching end of video. Restarting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lower the contour area to detect more individuals
    min_contour_area = 0
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Draw bounding boxes around the detected contours (people)
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display crowd density count on the frame
    cv2.putText(frame, f"Crowd Density: {len(large_contours)} people detected", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    # Set crowd detection threshold
    crowd_threshold = 20
    if len(large_contours) >= crowd_threshold:
        # Show alert if crowd exceeds threshold
        cv2.putText(frame, "High Alert: High Crowd Density!", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        print("🚨 ALERT: High Crowd Density Detected! 🚨")

        # Play sound alert only if cooldown time has passed
        current_time = time.time()
        if current_time - last_alert_time > alert_cooldown:
            pygame.mixer.music.load(alarm_sound)
            pygame.mixer.music.play()
            last_alert_time = current_time  # Update last alert time

    # Display the video frame with detection results
    cv2.imshow("Crowd Detection", frame)

    # Press 'q' to exit the loop and stop the video
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Release resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
