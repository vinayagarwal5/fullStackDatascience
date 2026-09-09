import speech_recognition as sr
import pyttsx3 as pt
import pywhatkit as pk

listening = sr.Recognizer()
engine = pt.init()   # ✅ fixed


def speak(text):
    engine.say(text)
    engine.runAndWait()

def hear():
    cmd = ""
    try:
        with sr.Microphone() as mic:
            listening.adjust_for_ambient_noise(mic, duration=1)
            print("Listening...")
            voice = listening.listen(mic, timeout=5, phrase_time_limit=8)
            cmd = listening.recognize_google(voice)
            cmd = cmd.lower()

            if "vinay" in cmd:
                cmd = cmd.replace("vinay", "")
                print(cmd)

    except sr.WaitTimeoutError:
        print("No speech detected. Please try again.")
    except sr.UnknownValueError:
        print("I could not understand the audio. Please speak clearly and try again.")
    except sr.RequestError as error:
        print(f"Speech recognition service error: {error}")
    except OSError as error:
        print(f"Microphone error: {error}")

    return cmd

def run():
    cmd = hear()

    if "play" in cmd:
        song = cmd.replace("play", "")
        speak("Playing " + song)
        pk.playonyt(song)   # ✅ fixed
run()