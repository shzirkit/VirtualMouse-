import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hands and pyautogui
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Variables to track state
prev_time = 0
clicking = False
dragging = False
action_text = "Move your hand"

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror-like effect
    img = cv2.flip(img, 1)

    # Convert the image to RGB as mediapipe uses RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates for the index finger tip (landmark 8)
            finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert to screen coordinates
            finger_x = int(finger_tip.x * screen_width)
            finger_y = int(finger_tip.y * screen_height)
            thumb_x, thumb_y = int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height)
            middle_x, middle_y = int(middle_tip.x * screen_width), int(middle_tip.y * screen_height)

            # Move the mouse
            pyautogui.moveTo(finger_x, finger_y)

            # Measure distances between fingertips
            distance_thumb_index = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([finger_x, finger_y]))
            distance_index_middle = np.linalg.norm(np.array([middle_x, middle_y]) - np.array([finger_x, finger_y]))

            # Define thresholds for gestures
            click_threshold = 50
            right_click_threshold = 50
            double_click_threshold = 50

            # Perform left click
            if distance_thumb_index < click_threshold and not clicking:
                pyautogui.click()
                clicking = True
                action_text = "Left Click"
                time.sleep(0.3)  # Avoid rapid clicking
            elif distance_thumb_index >= click_threshold:
                clicking = False

            # Perform right click
            if distance_index_middle < right_click_threshold:
                pyautogui.click(button='right')
                action_text = "Right Click"
                time.sleep(0.3)  # Avoid rapid right-clicking

            # Perform double click
            if distance_thumb_index < double_click_threshold and distance_index_middle < double_click_threshold:
                current_time = time.time()
                if current_time - prev_time < 0.5:  # If two clicks within 0.5 seconds, it's a double-click
                    pyautogui.doubleClick()
                    action_text = "Double Click"
                    time.sleep(0.3)  # Avoid rapid double-clicking
                prev_time = current_time

            # Perform drag
            if abs(finger_tip.y - finger_mcp.y) < 0.03:  # If the finger is straight (less vertical difference)
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                    action_text = "Dragging"
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

    else:
        action_text = "No hand detected"

    # Display the current action on the screen
    cv2.putText(img, f"Action: {action_text}", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the title "FAZMIC Mushaf Virtual Mouse" on the screen
    cv2.putText(img, "FAZMIC Mushaf Virtual Mouse", (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the webcam feed
    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
