import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])  # First point
    b = np.array([b.x, b.y])  # Middle point
    c = np.array([c.x, c.y])  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Initialize variables
reps = 0
points = 0
stage = None
exercise = "Squats"  # Default exercise
fullscreen = False  # Track fullscreen mode
level = 1
progress = 0
max_points = 50  # Points required to level up
achievements = []
history=[]

# Enhanced data tracking structure
data_tracking = {
    "Squats": {"times": [], "reps": [], "points": []},
    "Finger Twirling": {"times": [], "reps": [], "points": []},
    "Head Rotation": {"times": [], "reps": [], "points": []},
    "Fist Rotation": {"times": [], "reps": [], "points": []}
}

# Initialize tracking variables
time_start = datetime.now()
tracking_interval = 5  # Record data every 5 seconds

# Start video capture
cap = cv2.VideoCapture(0)
screen_width = 1920  # Replace with your screen width
screen_height = 1080  # Replace with your screen height

cv2.namedWindow("Exercise Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Exercise Tracker", 1280, 720)

print("Press 'Q' to quit. Use keys 1-4 to switch exercises.")
print("1: Squats, 2: Finger Twirling, 3: Head Rotation, 4: Fist Rotation")
print("Press 'F' to toggle fullscreen.")
print("Press 'G' to view progress graphs.")
def make_prediction(history):
    if len(history) < 5:
        return "Not enough data to predict."
    trends = np.diff(history[-5:])
    avg_trend = np.mean(trends)
    if avg_trend > 0:
        return "You're improving! Keep going!"
    elif avg_trend < 0:
        return "Your performance is declining. Try to focus!"
    else:
        return "Your progress is stable. Keep it up!"
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame. Exiting.")
        break

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    # Display selected exercise
    cv2.putText(frame, f"Exercise: {exercise}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Squat Tracking
    if exercise == "Squats" and results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results_pose.pose_landmarks.landmark

        left_hip = mp_pose.PoseLandmark.LEFT_HIP
        left_knee = mp_pose.PoseLandmark.LEFT_KNEE
        left_ankle = mp_pose.PoseLandmark.LEFT_ANKLE

        # Calculate knee angle
        knee_angle = calculate_angle(landmarks[left_hip], landmarks[left_knee], landmarks[left_ankle])

        # Detect the squat movement
        if knee_angle > 160:  # Standing
            stage = "up"
        elif knee_angle < 90 and stage == "up":  # Squatted
            stage = "down"
            reps += 1
            points += 10  # Award points for each squat

    # Finger Twirling Tracking
    elif exercise == "Finger Twirling" and results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate distance between thumb and index tips
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            if distance < 0.03:  # Threshold for closed position
                stage = "closed"
            elif distance > 0.06 and stage == "closed":  # Full twirl
                stage = "open"
                reps += 1
                points += 5  # Award points for each twirl

    # Head Rotation Tracking
    elif exercise == "Head Rotation" and results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results_pose.pose_landmarks.landmark

        left_ear = mp_pose.PoseLandmark.LEFT_EAR
        nose = mp_pose.PoseLandmark.NOSE
        right_ear = mp_pose.PoseLandmark.RIGHT_EAR

        # Calculate head rotation angle
        angle = calculate_angle(landmarks[left_ear], landmarks[nose], landmarks[right_ear])

        if angle > 140:  # Rotate fully to one side
            stage = "rotated"
        elif angle < 100 and stage == "rotated":  # Return to neutral
            stage = "neutral"
            reps += 1
            points += 7  # Award points for each full rotation

    # Fist Rotation Tracking
    elif exercise == "Fist Rotation" and results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate vertical movement for fist rotation
            distance = abs(wrist.y - pinky_tip.y)
            if distance > 0.1:  # Threshold for fist down
                stage = "down"
            elif distance < 0.05 and stage == "down":  # Fist up
                stage = "up"
                reps += 1
                points += 8  # Award points for each fist rotation

    # Track levels and achievements
    progress = (points % max_points) / max_points * 100
    if points // max_points + 1 > level:
        level += 1
        achievements.append(f"Reached Level {level}")

    # Track exercise data at regular intervals
    elapsed_time = datetime.now() - time_start
    if elapsed_time.total_seconds() >= tracking_interval:
        current_time = datetime.now()
        data_tracking[exercise]["times"].append(current_time)
        data_tracking[exercise]["reps"].append(reps)
        data_tracking[exercise]["points"].append(points)
        time_start = current_time
        history.append(points)
    prediction = make_prediction(history)

    # Display feedback
    cv2.putText(frame, f"Reps: {reps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Points: {points}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Level: {level}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(frame, (10, 200), (310, 230), (200, 200, 200), -1)
    cv2.rectangle(frame, (10, 200), (10 + int(300 * progress / 100), 230), (0, 255, 0), -1)
    cv2.putText(frame, f"Progress: {int(progress)}%", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if achievements:
        cv2.putText(frame, f"Achievements: {', '.join(achievements[-3:])}", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Resize and crop frame for fullscreen handling
    h, w, _ = frame.shape
    scale_width = screen_width / w
    scale_height = screen_height / h
    scale = max(scale_width, scale_height)

    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    start_x = (new_width - screen_width) // 2
    start_y = (new_height - screen_height) // 2
    cropped_frame = resized_frame[start_y:start_y + screen_height, start_x:start_x + screen_width]

    if fullscreen:
        cv2.imshow("Exercise Tracker", cropped_frame)
    else:
        cv2.imshow("Exercise Tracker", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('1'):
        exercise = "Squats"
        reps, points = 0, 0
    elif key == ord('2'):
        exercise = "Finger Twirling"
        reps, points = 0, 0
    elif key == ord('3'):
        exercise = "Head Rotation"
        reps, points = 0, 0
    elif key == ord('4'):
        exercise = "Fist Rotation"
        reps, points = 0, 0
    elif key == ord('f'):  # Toggle fullscreen mode
        fullscreen = not fullscreen
    elif key == ord('g'):  # Plot graphs
        if len(data_tracking[exercise]["times"]) > 0:
            plt.figure(figsize=(12, 6))
            
            # Convert timestamps to seconds elapsed
            start_time = data_tracking[exercise]["times"][0]
            times = [(t - start_time).total_seconds() for t in data_tracking[exercise]["times"]]
            
            # Plot both reps and points
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot reps
            ax1.plot(times, data_tracking[exercise]["reps"], 
                    marker='o', color='blue', label='Repetitions')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Repetitions')
            ax1.set_title(f'{exercise} - Repetitions Over Time')
            ax1.grid(True)
            ax1.legend()
            
            # Plot points
            ax2.plot(times, data_tracking[exercise]["points"], 
                    marker='s', color='green', label='Points')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Points')
            ax2.set_title(f'{exercise} - Points Over Time')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
        else:
            print("No data available for plotting yet. Perform some exercises first!")

cap.release()
cv2.destroyAllWindows() 
