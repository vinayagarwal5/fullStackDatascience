import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam Capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Read initial frame to dynamically match dimensions
ret, frame = cap.read()
if not ret:
    h, w, c = 720, 1280, 3
else:
    h, w, c = frame.shape

canvas = np.zeros((h, w, 3), np.uint8)

# 🎨 Color Configuration System (BGR format)
colors = [
    (255, 0, 255),  # Purple
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0)   # Cyan
]
draw_color = colors[0]  # Default: Purple
current_tool = "brush"  # Options: "brush", "eraser"

# 📏 Brush Settings
brush_thickness = 8
eraser_thickness = 50

# Layout Dimensions for Top Toolbar
header_h = 100
box_w = w // 6  # Split screen into 6 functional columns

# Previous tracking point
xp, yp = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror the frame

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if lm_list:
                x1, y1 = lm_list[8]   # Index finger tip
                x2, y2 = lm_list[12]  # Middle finger tip
                x0, y0 = lm_list[4]   # Thumb tip
                x4, y4 = lm_list[20]  # Pinky tip

                # 📐 Check which fingers are up
                fingers = []
                for tip_id in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky
                    fingers.append(lm_list[tip_id][1] < lm_list[tip_id - 2][1])

                # 🔍 1. DYNAMIC BRUSH RESIZER (Pinch Thumb and Pinky)
                # Calculates distance between thumb and pinky tips to dynamically set brush scale
                control_dist = int(np.hypot(x4 - x0, y4 - y0))
                if control_dist < 40:  # If pinch is detected, scale the size based on hand height
                    brush_thickness = max(2, min(50, int(np.hypot(x1 - x0, y1 - y0) // 3)))
                    cv2.putText(frame, f"Resizing Brush: {brush_thickness}px", (w - 300, h - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 👆 2. SELECTION MODE (Index + Middle Up)
                if fingers[0] and fingers[1]:
                    xp, yp = 0, 0  # Pause drawing
                    cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), (255, 255, 255), cv2.FILLED)

                    # Check if hovering over the top menu bar
                    if y1 < header_h:
                        col_idx = x1 // box_w
                        if col_idx < 4:  # Selected a color box
                            draw_color = colors[col_idx]
                            current_tool = "brush"
                        elif col_idx == 4:  # 🧽 3. ERASER TOOL MODE
                            current_tool = "eraser"
                        elif col_idx == 5:  # Global Canvas Clear
                            canvas = np.zeros_like(frame)

                # 🖋️ 4. DRAWING MODE (Only Index Up)
                elif fingers[0] and not fingers[1]:
                    cv2.circle(frame, (x1, y1), brush_thickness if current_tool == "brush" else 15, draw_color, -1)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if current_tool == "eraser":
                        cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), eraser_thickness)
                    else:
                        cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                    
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # 🖼️ RENDER TOP INTERACTIVE TOOLBAR MENU GUI
    labels = ["Purple", "Green", "Red", "Cyan"]
    for i in range(4):
        cv2.rectangle(frame, (i * box_w, 0), ((i + 1) * box_w, header_h), colors[i], cv2.FILLED)
        cv2.putText(frame, labels[i], (i * box_w + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Highlight chosen color box
        if draw_color == colors[i] and current_tool == "brush":
            cv2.rectangle(frame, (i * box_w, 0), ((i + 1) * box_w, header_h), (255, 255, 255), 4)

    # Render Eraser Box Interface
    cv2.rectangle(frame, (4 * box_w, 0), (5 * box_w, header_h), (40, 40, 40), cv2.FILLED)
    cv2.putText(frame, "ERASER", (4 * box_w + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if current_tool == "eraser":
        cv2.rectangle(frame, (4 * box_w, 0), (5 * box_w, header_h), (0, 255, 255), 4)

    # Render Reset Clear Box Interface
    cv2.rectangle(frame, (5 * box_w, 0), (w, header_h), (0, 0, 100), cv2.FILLED)
    cv2.putText(frame, "CLEAR ALL", (5 * box_w + 15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 🎛️ Blending Matrix Operations
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    frame_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    final = cv2.add(frame_bg, frame_fg)

    # Render Status Bar HUD info overlay
    cv2.putText(final, f"Tool: {current_tool.upper()} | Size: {brush_thickness if current_tool=='brush' else eraser_thickness}px", 
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Virtual Painter Pro", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()