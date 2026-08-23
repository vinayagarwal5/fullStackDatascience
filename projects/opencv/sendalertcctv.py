import cv2
import requests

TOKEN = "8794503737:AAFf-Fia_w3AOtRhkrSf_BP_AT5SjtQQOL4"
CHAT_ID = "6828570283"

def send_alert(frame=None):
    # 1. Send the text notification
    text_url = f"https://telegram.org{8794503737:AAFf-Fia_w3AOtRhkrSf_BP_AT5SjtQQOL4}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": "⚠️ Motion detected at your camera!"}
    requests.post(text_url, data=data)
    
    # 2. Send the image snapshot (if a frame is passed)
    if frame is not None:
        image_url = f"https://telegram.org{8794503737:AAFf-Fia_w3AOtRhkrSf_BP_AT5SjtQQOL4}/sendPhoto"
        
        # Convert the OpenCV frame to a JPEG in memory
        success, encoded_image = cv2.imencode('.jpg', frame)
        if success:
            files = {'photo': ('alert.jpg', encoded_image.tobytes(), 'image/jpeg')}
            requests.post(image_url, data={'chat_id': CHAT_ID}, files=files)