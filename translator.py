import cv2
import numpy as np
import speech_recognition as sr
from googletrans import Translator
import mediapipe as mp
from gtts import gTTS
import os
import subprocess
from collections import deque

# Initialize modules
translator = Translator()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Enhanced sign language gestures (ASL alphabet)
GESTURES = {
    tuple([0,0,0,0,0]): "A",
    tuple([1,1,0,0,0]): "B",
    tuple([1,0,0,0,0]): "C",
    tuple([0,1,1,1,1]): "D",
    tuple([1,1,1,1,1]): "E",
    tuple([0,1,1,1,0]): "F",
    tuple([1,1,1,0,0]): "G",
    tuple([0,1,0,0,0]): "H",
    tuple([0,0,0,0,1]): "I",
    tuple([0,1,1,1,1]): "J",
    tuple([1,0,0,0,1]): "K",
    tuple([1,1,0,0,1]): "L",
    tuple([0,0,0,0,0]): "M",
    tuple([0,0,0,0,0]): "N",
    tuple([0,0,0,0,0]): "O",
    tuple([1,1,1,1,1]): "Hello",
    tuple([0,1,0,0,0]): "Help",
    tuple([0,0,0,0,1]): "I",
    tuple([1,1,0,0,1]): "Love",
    tuple([0,0,0,0,0]): "You",
}

def speak_text(text, lang='en'):
    """Convert text to speech using gTTS without file permission issues"""
    try:
        # Create temporary file in user's temp directory
        temp_file = os.path.join(os.environ['TEMP'], 'temp_translation.mp3')
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_file)
        
        # Play using system default player
        if os.name == 'nt':  # Windows
            os.startfile(temp_file)
        else:  # Mac/Linux
            subprocess.call(['open', temp_file] if sys.platform == 'darwin' else ['xdg-open', temp_file])
            
    except Exception as e:
        print(f"Speech error: {e}")
        print(f"(Audio failed) Translation: {text}")

def continuous_listen():
    """Listen continuously to microphone input"""
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8
    recognizer.energy_threshold = 4000
    
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready to listen...")
        
        while True:
            try:
                print("\nSpeak now (or say 'mode' to switch):")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("Could not understand audio")
                continue
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                continue

def translate_text(text, target_lang='en'):
    """Translate text to target language"""
    try:
        translation = translator.translate(text, dest=target_lang)
        print(f"Translation: {translation.text}")
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def detect_gesture(frame):
    """Detect hand gestures from camera frame"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    gesture_text = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            finger_states = []
            for finger_tip in [8, 12, 16, 20]:  # Finger tip landmarks
                if hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_tip-2].y:
                    finger_states.append(1)
                else:
                    finger_states.append(0)
            
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                finger_states.append(1)
            else:
                finger_states.append(0)
            
            gesture_key = tuple(finger_states)
            if gesture_key in GESTURES:
                gesture_text = GESTURES[gesture_key]
                cv2.putText(frame, gesture_text, (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame, gesture_text

def sign_language_mode():
    """Sign language recognition mode"""
    cap = cv2.VideoCapture(0)
    gesture_history = deque(maxlen=5)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame, gesture_text = detect_gesture(frame)
        
        if gesture_text:
            gesture_history.append(gesture_text)
            if gesture_history.count(gesture_text) >= 3:
                print(f"Sign detected: {gesture_text}")
                translated = translate_text(gesture_text)
                speak_text(translated)
                gesture_history.clear()
        
        cv2.imshow('Sign Language Translator', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def speech_translation_mode(target_lang='fr'):
    """Continuous speech translation mode"""
    print(f"\nTranslating to: {target_lang} (Press Ctrl+C to exit)")
    
    while True:
        try:
            text = continuous_listen()
            if text and "mode" in text.lower():
                break
                
            if text:
                translated = translate_text(text, target_lang)
                speak_text(translated, target_lang)
                
        except KeyboardInterrupt:
            break

def main():
    """Main program"""
    print("=== Live Language & Sign Language Translator ===")
    
    while True:
        print("\nSelect mode:")
        print("1. Speech Translation")
        print("2. Sign Language Translation")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            lang = input("Enter target language code (e.g., 'fr' for French): ").strip()
            speech_translation_mode(lang)
        elif choice == '2':
            sign_language_mode()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()