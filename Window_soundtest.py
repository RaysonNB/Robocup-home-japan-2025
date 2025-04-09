from playsound import playsound
import pyttsx3
import speech_recognition as sr
import time
def speak(message):
    engine = pyttsx3.init()

    # Speak the message
    print("Speaking...")
    engine.say(message)
    engine.runAndWait()


def recognize_speech(duration):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio_data = recognizer.record(source, duration=duration)
        try:
            return recognizer.recognize_google(audio_data).lower()
        except sr.UnknownValueError:
            return None


def confirm_recording(original_text):
    while True:  # Keep asking until we get a "yes"
        speak(f"Did you say: {original_text}?")
        speak("Please answer hello fambot 'yes' or 'no' after the sound")
        time.sleep(1)
        playsound("nigga2.mp3")
        response = recognize_speech(5)

        if response is None:
            speak("Sorry, I didn't catch that. Let's try again.")
            continue

        print(f"You responded: {response}")

        if "yes" in response:
            return True
        elif "no" in response:
            return False
        else:
            speak("I didn't understand your answer. Please say 'yes' or 'no'.")


def get_user_input(max_attempts=3):
    for attempt in range(max_attempts):
        # Voice introduction
        speak("Hi, I am fambot. How can I help you? Speak after the")
        playsound("nigga2.mp3")  # Changed sound file name
        speak("sound, please wait a few seconds to start")

        # Countdown
        time.sleep(1)
        playsound("nigga2.mp3")

        user_text = recognize_speech(10)

        if not user_text:
            error_msg = "Sorry, I couldn't understand. Please try again."
            print(error_msg)
            speak(error_msg)
            continue

        print(f"You said: {user_text}")
        #speak(f"You said: {user_text}")
        user_text=user_text.replace("facebook","fambot")
        if confirm_recording(user_text):
            return user_text
        else:
            speak("Let's try recording again.")

    return None


# Main execution
for i in range(3):
    user_input = get_user_input()
    if user_input:
        print(f"Final recognized input: {user_input},    {i}")
    else:
        speak("I couldn't understand your input after several attempts. Please try again later.")
