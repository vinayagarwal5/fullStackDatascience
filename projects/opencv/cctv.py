import cv2
import time

# 1. Connect to your DVR camera stream
# Replace with your actual RTSP URL. Use 0 instead of the string to test with a local webcam.
RTSP_URL = "rtsp://admin:admin%40123@192.168.31.250:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(RTSP_URL)

# Initialize variables for motion detection
ret, frame1 = cap.read()
ret, frame2 = cap.read()

last_alert_time = 0
ALERT_COOLDOWN = 10  # Seconds to wait between sending alerts

def send_alert():
    """Trigger your alert system here (e.g., Email, Telegram, or WhatsApp)"""
    print("[ALERT] Motion detected by DVR Camera!")
    # Example: You can integrate Twilio or a Discord webhook here.

while cap.isOpened():
    # Calculate the absolute difference between two consecutive frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        # Ignore small movements like bugs, dust, or shadows
        if cv2.contourArea(contour) < 5000:
            continue
        
        motion_detected = True
        
        # Draw bounding boxes around the moving object
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Trigger alert if motion is found and cooldown has passed
    if motion_detected and (time.time() - last_alert_time > ALERT_COOLDOWN):
        send_alert()
        last_alert_time = time.time()

    # Display the processed live feed
    cv2.imshow("DVR Smart Sensor Feed", frame1)
    
    # Update frames for the next iteration
    frame1 = frame2
    ret, frame2 = cap.read()

    # Press 'q' to exit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()