import os
import warnings

# 🤫 SILENCE INTERNAL LOGS: Must be executed BEFORE importing cv2 or mediapipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'       # Suppresses TensorFlow CPU/XNNPACK logging messages
os.environ['GLOG_minloglevel'] = '3'           # Suppresses C++ Google Log (absl) feedback warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf") # Suppresses SymbolDatabase warning

import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Open Webcam to catch hardware dimensions dynamically
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

success, img = cap.read()
if not success:
    h, w, c = 720, 1280, 3
else:
    h, w, c = img.shape

# Canvas layer holding the live drawings
img_canvas = np.zeros((h, w, 3), np.uint8)

# Custom color choices (BGR format matrix)
colors_list = [
    (255, 0, 255),  # Purple
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan 
    (0, 255, 255),  # Yellow
    (0, 165, 255)   # Orange
]
draw_color = colors_list[0]  # Default color: Purple

brush_thickness = 7
eraser_thickness = 50
xp, yp = 0, 0

# State tracking stack for multi-step undos
undo_stack = [img_canvas.copy()]

# Shape Selector settings
current_tool = 'brush' # Options: 'brush', 'circle', 'rectangle', 'eraser'

# UI Panel layout dimensions calculated dynamically
header_h = 110
num_boxes = 8 
box_w = w // num_boxes

# Finger tips tracking definitions
tip_ids = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks):
    fingers = []
    # Thumb (horizontal alignment check)
    fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    # 4 Fingers (vertical alignment check)
    for id in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y else 0)
    return fingers

tool_selected_last_frame = False

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Temporary mask overlays to draw shapes interactively before locking them to canvas
    img_temp_shape = np.zeros((h, w, 3), np.uint8)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if lm_list:
                x1, y1 = lm_list[8]   # Index fingertip
                x2, y2 = lm_list[12]  # Middle fingertip

                fingers = fingers_up(handLms)

                # ----------------- SELECTION MODE (Two Fingers Up) -----------------
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    cv2.rectangle(img, (x1, y1-25), (x2, y2+25), draw_color, cv2.FILLED)
                    
                    if y1 < header_h:
                        col_idx = x1 // box_w
                        
                        if col_idx < 6: # Custom Color Grid slots chosen
                            draw_color = colors_list[col_idx]
                            if current_tool == 'eraser':
                                current_tool = 'brush'
                        elif col_idx == 6: # Shapes & Eraser Selection Cycle Trigger Box
                            if not tool_selected_last_frame:
                                if current_tool == 'brush': current_tool = 'circle'
                                elif current_tool == 'circle': current_tool = 'rectangle'
                                elif current_tool == 'rectangle': current_tool = 'eraser'
                                elif current_tool == 'eraser': current_tool = 'brush'
                                tool_selected_last_frame = True
                    else:
                        tool_selected_last_frame = False

                # ----------------- ACTION / DRAWING MODE (Index Finger Up Only) -----------------
                elif fingers[1] and not fingers[2]:
                    tool_selected_last_frame = False
                    cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if current_tool == 'eraser':
                        cv2.line(img, (xp, yp), (x1, y1), (0, 0, 0), eraser_thickness)
                        cv2.line(img_canvas, (xp, yp), (x1, y1), (0, 0, 0), eraser_thickness)
                        xp, yp = x1, y1
                        
                    elif current_tool == 'brush':
                        cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                        cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                        xp, yp = x1, y1
                        
                    elif current_tool == 'circle':
                        radius = int(np.hypot(x1 - xp, y1 - yp))
                        cv2.circle(img_temp_shape, (xp, yp), radius, draw_color, brush_thickness)
                        
                    elif current_tool == 'rectangle':
                        cv2.rectangle(img_temp_shape, (xp, yp), (x1, y1), draw_color, brush_thickness)
                
                # ----------------- SHAPE RELEASE GESTURE -----------------
                else:
                    tool_selected_last_frame = False
                    if current_tool in ['circle', 'rectangle'] and xp != 0 and yp != 0:
                        if current_tool == 'circle':
                            radius = int(np.hypot(x1 - xp, y1 - yp))
                            cv2.circle(img_canvas, (xp, yp), radius, draw_color, brush_thickness)
                        elif current_tool == 'rectangle':
                            cv2.rectangle(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                        
                        undo_stack.append(img_canvas.copy())
                        if len(undo_stack) > 20: 
                            undo_stack.pop(0)
                            
                    xp, yp = 0, 0
                    
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
    else:
        if current_tool in ['circle', 'rectangle'] and xp != 0 and yp != 0:
            if current_tool == 'circle':
                radius = int(np.hypot(x1 - xp, y1 - yp))
                cv2.circle(img_canvas, (xp, yp), radius, draw_color, brush_thickness)
            elif current_tool == 'rectangle':
                cv2.rectangle(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
            undo_stack.append(img_canvas.copy())
            if len(undo_stack) > 20:
                undo_stack.pop(0)
        xp, yp = 0, 0
        tool_selected_last_frame = False

    # Blending operations
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_and(img, img, mask=img_inv)
    img = cv2.bitwise_or(img, img_canvas)
    
    if np.sum(img_temp_shape) > 0:
        img = cv2.bitwise_or(img, img_temp_shape)

    # ----------------- RENDERING THE INTERACTIVE TOOLBAR -----------------
    labels = ["Purple", "Green", "Red", "Cyan", "Yellow", "Orange"]
    for i in range(6):
        cv2.rectangle(img, (i * box_w, 0), ((i + 1) * box_w, header_h), colors_list[i], cv2.FILLED)
        if draw_color == colors_list[i] and current_tool != 'eraser':
            cv2.rectangle(img, (i * box_w, 0), ((i + 1) * box_w, header_h), (255, 255, 255), 4)
        cv2.putText(img, labels[i], (i * box_w + 10, header_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Tool Selection Box
    cv2.rectangle(img, (6 * box_w, 0), (7 * box_w, header_h), (50, 50, 50), cv2.FILLED)
    if current_tool == 'eraser':
        cv2.rectangle(img, (6 * box_w, 0), (7 * box_w, header_h), (255, 255, 255), 4)
    cv2.putText(img, f"TOOL:", (6 * box_w + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, current_tool.upper(), (6 * box_w + 10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # System Status Monitor Box
    cv2.rectangle(img, (7 * box_w, 0), (w, header_h), (25, 25, 25), cv2.FILLED)
    cv2.putText(img, "U: Undo", (7 * box_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "S: Save Image", (7 * box_w + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "ESC: Close", (7 * box_w + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Bottom UI Status Bar
    cv2.rectangle(img, (0, h - 40), (w, h), (0, 0, 0), cv2.FILLED)
    status_text = f"Active Config: Tool -> {current_tool.upper()} | Canvas States Stack Size -> {len(undo_stack)}"
    cv2.putText(img, status_text, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if current_tool in ['brush', 'eraser'] and results.multi_hand_landmarks and (fingers[1] and not fingers[2]):
        pass
    elif current_tool in ['brush', 'eraser'] and xp == 0 and yp == 0:
        if not np.array_equal(img_canvas, undo_stack[-1]):
            undo_stack.append(img_canvas.copy())
            if len(undo_stack) > 20:
                undo_stack.pop(0)

    cv2.imshow("Advanced Air Writing & Canvas Studio", img)

    key = cv2.waitKey(1)
    if key == ord('u') or key == ord('U'): 
        if len(undo_stack) > 1:
            undo_stack.pop() 
            img_canvas = undo_stack[-1].copy() 
            print("Undo executed successfully!")
        else:
            img_canvas = np.zeros((h, w, 3), np.uint8)
            undo_stack = [img_canvas.copy()]
    elif key == ord('s') or key == ord('S'):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"air_drawing_{timestamp}.png"
        cv2.imwrite(filename, img_canvas)
        print(f"Canvas saved successfully as {filename}!")