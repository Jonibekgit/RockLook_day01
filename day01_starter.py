
import cv2
import mediapipe as mp
import pygame
import sys
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)  # Try second camera index
if not cap.isOpened():
    print("ERROR: No webcam found. Check your camera connection.")
    sys.exit(1)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize audio
pygame.mixer.init()

# Load your music file — put any .mp3 in the same folder
MUSIC_FILE = "music.mp3"
if os.path.exists(MUSIC_FILE):
    pygame.mixer.music.load(MUSIC_FILE)
    print(f"Loaded: {MUSIC_FILE}")
else:
    print(f"WARNING: '{MUSIC_FILE}' not found. Place an .mp3 file in this folder.")
    print("The program will still run, but no music will play.")

LEFT_IRIS = 468
RIGHT_IRIS = 473

# Nose tip (used as face center reference)
NOSE_TIP = 1

is_playing = False

GAZE_THRESHOLD = 0.02 


# ============================================================
# MAIN LOOP — Sensor → Process → Output
# ============================================================
print("\nRockLook is running!")
print(f"Threshold: {GAZE_THRESHOLD}")
print("Look DOWN to play music. Look UP to pause.")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame so it mirrors your movement (like a mirror)
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe needs RGB, OpenCV gives BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get iris and nose positions (normalized 0-1)
        left_iris_y = landmarks[LEFT_IRIS].y
        right_iris_y = landmarks[RIGHT_IRIS].y
        nose_y = landmarks[NOSE_TIP].y

        # Average iris position
        iris_y = (left_iris_y + right_iris_y) / 2

        # Calculate how far the iris is below the nose
        # Positive = looking down, Negative = looking up
        gaze_offset = iris_y - nose_y

        # Display the gaze value on screen
        looking_down = gaze_offset > GAZE_THRESHOLD
        status = "LOOKING DOWN" if looking_down else "LOOKING UP"
        color = (0, 0, 255) if looking_down else (0, 255, 0)

        cv2.putText(frame, f"Gaze offset: {gaze_offset:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {GAZE_THRESHOLD}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        if looking_down and not is_playing:
            if os.path.exists(MUSIC_FILE):
                if pygame.mixer.music.get_pos() == -1:
                    pygame.mixer.music.play()
                else:
                    pygame.mixer.music.unpause()
                is_playing = True
                print("▶ Music playing")
        elif not looking_down and is_playing:
            pygame.mixer.music.pause()
            is_playing = False
            print("⏸ Music paused")

    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the webcam feed
    cv2.imshow("RockLook - Day 01", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("\nRockLook ended. See you tomorrow for Day 02!")
