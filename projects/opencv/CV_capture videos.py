import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Deque-like list to store all the points where the red object has moved
points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the frame horizontally so it acts like a mirror (intuitive for drawing)
    frame = cv2.flip(frame, 1)
    
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define Red Ranges (using the dual-mask logic from earlier)
    low_red1 = np.array([0, 120, 70])
    high_red1 = np.array([10, 255, 255])
    low_red2 = np.array([170, 120, 70])
    high_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_frame, low_red1, high_red1)
    mask2 = cv2.inRange(hsv_frame, low_red2, high_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up the mask noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center = None
    
    if len(contours) > 0:
        # Find the largest contour (the main red object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 500:
            # Calculate the minimum enclosing circle around the red object
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            
            # Calculate the exact center point (centroid) of the object
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
                # Draw a pointer dot on the screen where the object is
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
    
    # Append the current position to our point history
    if center is not None:
        points.append(center)
    else:
        # Adding None creates a break in the drawing if you hide the object
        points.append(None)
        
    # Draw the continuous canvas lines by connecting historical points
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
            
        # Draw a thick blue line between consecutive tracking points
        cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 4)
        
    # Instructions displayed on screen
    cv2.putText(frame, "Move RED object to paint. Press 'c' to clear.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Air Canvas", frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('c'):  # Press 'c' to clear the canvas
        points = []

cap.release()
cv2.destroyAllWindows()
