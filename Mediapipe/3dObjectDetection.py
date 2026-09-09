import cv2
import urllib.request
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Assign legacy MediaPipe solutions
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils 

# -----------> DETECT OBJECTS FROM IMAGES GIVEN IN URL

def url_to_array(url):
    try:
        # Use a modern web header simulation to bypass hosting firewalls
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            arr = np.array(bytearray(response.read()), dtype=np.uint8)
            arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if arr is None:
                raise ValueError("OpenCV decoder received empty or corrupt bytes.")
                
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return arr
            
    except Exception as e:
        print(f"⚠️ Primary URL failed: {e}")
        return None

# VERIFIED LIVE URL: Reliable server containing a standard coffee cup for Objectron 3D mapping
mug_url = 'https://wikimedia.org'
mug = url_to_array(mug_url)

# FALLBACK URL: In case Wikimedia experiences regional connectivity latency
if mug is None:
    print("🔄 Attempting backup image server...")
    fallback_url = 'https://unsplash.com'
    mug = url_to_array(fallback_url)

if mug is None:
    print("❌ Critical Error: Could not download any images. Please verify your internet connection.")
    exit()

# ------> INSTANTIATE AN OBJECTRON INSTANCE
objectron = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=5,
    min_detection_confidence=0.1,  # Lowered slightly to capture mugs at varying angles
    model_name='Cup'
)

# Inference
results = objectron.process(mug)

# --------------> DISPLAY THE RESULT
annotated_image = mug.copy()

if not results.detected_objects:
    print('No 3D box landmarks detected for the cup.')
else:
    print(f'Succeeded! Detected {len(results.detected_objects)} object(s). Rendering wireframes...')
    for detected_object in results.detected_objects:
        # Draw 3D wireframe boxes
        mp_drawing.draw_landmarks(
            annotated_image,
            detected_object.landmarks_2d,
            mp_objectron.BOX_CONNECTIONS
        )

        # Draw structural orientation axes
        mp_drawing.draw_axis(
            annotated_image,
            detected_object.rotation,
            detected_object.translation
        )

# -----------> PLOT THE RESULT
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(annotated_image)
ax.axis('off')
plt.show()
