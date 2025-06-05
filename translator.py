import cv2
import numpy as np
import os
import subprocess
import sys
import threading
import time
from collections import deque, Counter

# Check for required dependencies with alternatives
missing_deps = []

# Try different audio libraries for speech recognition
SPEECH_AVAILABLE = False
try:
    import speech_recognition as sr
    # Try PyAudio first
    try:
        import pyaudio
        SPEECH_AVAILABLE = True
        AUDIO_BACKEND = "pyaudio"
        print("✅ Using PyAudio for speech recognition")
    except ImportError:
        # Try alternative: use system microphone without PyAudio
        try:
            # Test if we can create a recognizer without PyAudio
            recognizer = sr.Recognizer()
            # This will work on some systems with built-in audio
            SPEECH_AVAILABLE = True
            AUDIO_BACKEND = "system"
            print("✅ Using system audio for speech recognition")
        except Exception:
            SPEECH_AVAILABLE = False
            AUDIO_BACKEND = None
            print("⚠️ No audio backend available")
except ImportError as e:
    SPEECH_AVAILABLE = False
    AUDIO_BACKEND = None
    missing_deps.append("speech_recognition")
    print("⚠️ Speech recognition not available. Install with: pip install SpeechRecognition")

try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    missing_deps.append("googletrans")
    print("⚠️ Translation not available. Install with: pip install googletrans==4.0.0rc1")

try:
    import mediapipe as mp
    SIGN_LANGUAGE_AVAILABLE = True
except ImportError:
    SIGN_LANGUAGE_AVAILABLE = False
    missing_deps.append("mediapipe")
    print("⚠️ Sign language recognition not available. Install with: pip install mediapipe")

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    missing_deps.append("gtts")
    print("⚠️ Text-to-speech not available. Install with: pip install gtts")

# Initialize modules only if available
translator = None
mp_hands = None
hands = None
mp_drawing = None

if TRANSLATION_AVAILABLE:
    try:
        translator = Translator()
        print("✅ Translation service initialized")
    except Exception as e:
        print(f"⚠️ Translation service failed to initialize: {e}")
        TRANSLATION_AVAILABLE = False

if SIGN_LANGUAGE_AVAILABLE:
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, 
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        print("✅ Sign language recognition initialized")
    except Exception as e:
        print(f"⚠️ Sign language recognition failed to initialize: {e}")
        SIGN_LANGUAGE_AVAILABLE = False

# Enhanced sign language gestures (ASL alphabet) - Fixed mappings
GESTURES = {
    # Basic letters
    tuple([0,0,0,0,0]): "A",
    tuple([1,1,1,1,0]): "B", 
    tuple([1,0,0,0,1]): "C",
    tuple([0,1,1,1,1]): "D",
    tuple([0,0,0,0,0]): "E",  # Fist
    tuple([0,1,1,1,0]): "F",
    tuple([1,1,0,0,1]): "G",
    tuple([0,1,1,0,0]): "H",
    tuple([0,0,0,0,1]): "I",
    tuple([0,0,0,0,1]): "J",  # Same as I with motion
    tuple([1,1,0,0,1]): "K",
    tuple([1,0,0,0,0]): "L",
    tuple([0,0,0,1,0]): "M",
    tuple([0,0,1,1,0]): "N",
    tuple([0,0,0,0,0]): "O",  # Circle shape
    tuple([1,1,0,0,0]): "P",
    tuple([1,1,0,0,1]): "Q",
    tuple([0,1,1,0,0]): "R",
    tuple([0,0,0,1,1]): "S",
    tuple([0,0,0,0,0]): "T",  # Thumb between fingers
    tuple([0,0,1,1,0]): "U",
    tuple([0,1,1,0,0]): "V",
    tuple([0,1,1,1,0]): "W",
    tuple([1,0,0,0,0]): "X",
    tuple([0,0,0,0,1]): "Y",
    tuple([1,0,0,0,0]): "Z",
    
    # Common words/phrases
    tuple([1,1,1,1,1]): "Hello",
    tuple([0,1,0,0,0]): "Help",
    tuple([1,0,1,0,1]): "Love",
    tuple([0,1,1,1,0]): "Thank you",
    tuple([1,0,0,1,0]): "Yes",
    tuple([0,0,1,0,0]): "No",
    tuple([1,1,0,1,1]): "Please",
    tuple([0,0,0,1,0]): "Sorry",
}

def speak_text(text, lang='en'):
    """Convert text to speech using gTTS with better error handling"""
    if not TTS_AVAILABLE:
        print(f"🔊 Would speak: {text}")
        return
        
    try:
        if not text or text.strip() == "":
            return
            
        print(f"Speaking: {text}")
        
        # Create temporary file with better path handling
        temp_dir = os.path.join(os.path.expanduser("~"), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f'translation_{int(time.time())}.mp3')
        
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(temp_file)
        
        # Play using system default player
        if os.name == 'nt':  # Windows
            os.startfile(temp_file)
        elif sys.platform == 'darwin':  # Mac
            subprocess.call(['open', temp_file])
        else:  # Linux
            subprocess.call(['xdg-open', temp_file])
            
        # Clean up after a delay
        threading.Timer(5.0, lambda: os.remove(temp_file) if os.path.exists(temp_file) else None).start()
            
    except Exception as e:
        print(f"Speech error: {e}")
        print(f"(Audio failed) Translation: {text}")

def continuous_listen():
    """Listen continuously to microphone input with multiple backend support"""
    if not SPEECH_AVAILABLE:
        print("❌ Speech recognition not available. Please install required packages.")
        return None
        
    try:
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1.0
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        
        # Try different microphone initialization methods
        microphone = None
        
        if AUDIO_BACKEND == "pyaudio":
            try:
                microphone = sr.Microphone()
                print("🎤 Using PyAudio microphone")
            except Exception as e:
                print(f"PyAudio microphone failed: {e}")
                
        if microphone is None:
            # Try system default microphone without PyAudio
            try:
                microphone = sr.Microphone(device_index=None)
                print("🎤 Using system default microphone")
            except Exception as e:
                print(f"System microphone failed: {e}")
                # Try without specifying device
                try:
                    import speech_recognition as sr
                    microphone = sr.Microphone()
                    print("🎤 Using fallback microphone")
                except Exception as e2:
                    print(f"❌ All microphone methods failed: {e2}")
                    return None
        
        if microphone is None:
            print("❌ Could not initialize microphone")
            return None
            
        with microphone as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Ready to listen... (Speak clearly)")
            
            while True:
                try:
                    print("\n🎤 Listening...")
                    # Listen with longer timeout and phrase limit
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=8)
                    
                    print("🔄 Processing speech...")
                    text = recognizer.recognize_google(audio, language='en-US')
                    
                    if text and len(text.strip()) > 0:
                        print(f"✅ You said: '{text}'")
                        return text.strip()
                    
                except sr.WaitTimeoutError:
                    print("⏱️ No speech detected, continuing to listen...")
                    continue
                except sr.UnknownValueError:
                    print("❌ Could not understand audio, please speak clearly")
                    continue
                except sr.RequestError as e:
                    print(f"❌ Speech recognition service error: {e}")
                    time.sleep(1)
                    continue
                    
    except ImportError:
        print("❌ Audio libraries not available.")
        print("\n📦 Try these installation methods:")
        print("1. conda install -c conda-forge pyaudio")
        print("2. pip install --only-binary=all pyaudio")
        print("3. Use online speech recognition instead")
        return None
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        print("\n📋 Troubleshooting tips:")
        print("1. Check microphone permissions")
        print("2. Try: conda install -c conda-forge pyaudio")
        print("3. Use text input mode instead")
        return None

def translate_text(text, target_lang='en'):
    """Translate text to target language with better error handling"""
    if not TRANSLATION_AVAILABLE:
        print(f"🔄 Would translate: '{text}' to {target_lang}")
        return text
        
    if not text or text.strip() == "":
        return ""
        
    try:
        print(f"🔄 Translating '{text}' to {target_lang}...")
        translation = translator.translate(text, dest=target_lang)
        result = translation.text
        print(f"✅ Translation: '{result}'")
        return result
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return text

def get_finger_positions(landmarks):
    """Get finger positions more accurately"""
    finger_states = []
    
    # Thumb (different logic because it moves horizontally)
    if landmarks[4].x > landmarks[3].x:  # Thumb tip vs thumb IP
        finger_states.append(1)
    else:
        finger_states.append(0)
    
    # Other fingers (index, middle, ring, pinky)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:  # Tip above PIP = extended
            finger_states.append(1)
        else:
            finger_states.append(0)
    
    return finger_states

def detect_gesture(frame):
    """Detect hand gestures from camera frame with improved accuracy"""
    if not SIGN_LANGUAGE_AVAILABLE:
        cv2.putText(frame, "Sign language not available", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, None, 0
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    gesture_text = None
    confidence = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            # Get finger positions
            finger_states = get_finger_positions(hand_landmarks.landmark)
            gesture_key = tuple(finger_states)
            
            # Check for gesture match
            if gesture_key in GESTURES:
                gesture_text = GESTURES[gesture_key]
                confidence = 0.9
            else:
                # Try to find closest match
                best_match = None
                min_diff = float('inf')
                
                for key, value in GESTURES.items():
                    diff = sum(a != b for a, b in zip(gesture_key, key))
                    if diff < min_diff and diff <= 1:  # Allow 1 finger difference
                        min_diff = diff
                        best_match = value
                        confidence = 0.7 - (diff * 0.2)
                
                if best_match and confidence > 0.5:
                    gesture_text = best_match
            
            # Display gesture on frame
            if gesture_text:
                cv2.putText(frame, f"{gesture_text} ({confidence:.1f})", 
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Fingers: {finger_states}", 
                          (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame, gesture_text, confidence

def sign_language_mode():
    """Sign language recognition mode with improved detection"""
    if not SIGN_LANGUAGE_AVAILABLE:
        print("❌ Sign language recognition not available.")
        print("Install with: pip install mediapipe opencv-python")
        return
        
    print("\n🤟 Starting Sign Language Mode")
    print("Show your hand gestures to the camera")
    print("Press 'q' to quit, 's' to switch modes")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    gesture_history = deque(maxlen=10)
    last_spoken_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect gestures
        frame, gesture_text, confidence = detect_gesture(frame)
        
        # Add gesture to history if detected with good confidence
        if gesture_text and confidence > 0.7:
            gesture_history.append(gesture_text)
            
            # Check if we have consistent detection
            gesture_counts = Counter(gesture_history)
            most_common = gesture_counts.most_common(1)[0]
            
            if most_common[1] >= 5 and time.time() - last_spoken_time > 2:
                detected_gesture = most_common[0]
                print(f"✅ Sign detected: {detected_gesture}")
                
                # Translate and speak
                translated = translate_text(detected_gesture, 'en')
                speak_text(translated)
                
                gesture_history.clear()
                last_spoken_time = time.time()
        
        # Add instructions to frame
        cv2.putText(frame, "Press 'q' to quit, 's' to switch modes", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Sign Language Translator', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('s'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def text_input_mode(target_lang='es'):
    """Text-based translation mode as fallback"""
    print(f"\n⌨️ Text Translation Mode")
    print(f"Translating to {target_lang}")
    print("Type 'quit' to exit, 'switch' to change modes")
    
    while True:
        try:
            text = input("\n💬 Enter text to translate: ").strip()
            
            if not text:
                continue
                
            if text.lower() in ['quit', 'exit', 'switch', 'change mode']:
                break
                
            # Translate and speak
            translated = translate_text(text, target_lang)
            if translated and translated != text:
                speak_text(translated, target_lang)
            else:
                print("⚠️ Translation failed or text unchanged")
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error in text mode: {e}")

def speech_translation_mode(source_lang='auto', target_lang='es'):
    """Continuous speech translation mode with fallback to text input"""
    if not SPEECH_AVAILABLE:
        print("❌ Speech recognition not available.")
        print("🔄 Switching to text input mode...")
        text_input_mode(target_lang)
        return
        
    print(f"\n🎤 Starting Speech Translation Mode")
    print(f"Audio Backend: {AUDIO_BACKEND}")
    print(f"Translating from {source_lang} to {target_lang}")
    print("Say 'switch mode' or 'quit' to exit")
    print("If speech fails, we'll switch to text mode")
    
    speech_failures = 0
    max_failures = 3
    
    while True:
        try:
            text = continuous_listen()
            
            if not text:
                speech_failures += 1
                if speech_failures >= max_failures:
                    print(f"\n⚠️ Too many speech recognition failures ({speech_failures})")
                    print("🔄 Switching to text input mode...")
                    text_input_mode(target_lang)
                    break
                continue
            
            # Reset failure counter on success
            speech_failures = 0
                
            # Check for mode switching commands
            text_lower = text.lower()
            if any(phrase in text_lower for phrase in ['switch mode', 'change mode', 'quit', 'exit']):
                print("🔄 Switching modes...")
                break
            
            # Translate and speak
            if len(text.strip()) > 0:
                translated = translate_text(text, target_lang)
                if translated and translated != text:
                    speak_text(translated, target_lang)
                else:
                    print("⚠️ Translation failed or text unchanged")
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error in speech mode: {e}")
            speech_failures += 1
            if speech_failures >= max_failures:
                print("🔄 Switching to text input mode...")
                text_input_mode(target_lang)
                break
            time.sleep(1)

def get_language_code():
    """Get language code from user with validation"""
    languages = {
        'spanish': 'es', 'french': 'fr', 'german': 'de', 'italian': 'it',
        'portuguese': 'pt', 'chinese': 'zh', 'japanese': 'ja', 'korean': 'ko',
        'russian': 'ru', 'arabic': 'ar', 'hindi': 'hi', 'english': 'en'
    }
    
    print("\nAvailable languages:")
    for lang_name, code in languages.items():
        print(f"  {lang_name.capitalize()}: {code}")
    
    while True:
        lang_input = input("\nEnter language name or code: ").strip().lower()
        
        if lang_input in languages:
            return languages[lang_input]
        elif lang_input in languages.values():
            return lang_input
        else:
            print("❌ Invalid language. Please try again.")

def main():
    """Main program with improved interface"""
    print("=" * 50)
    print("🌍 Live Language & Sign Language Translator")
    print("=" * 50)
    
    # Check availability status
    print("\n📋 System Status:")
    print(f"🎤 Speech Recognition: {'✅ Available' if SPEECH_AVAILABLE else '❌ Not Available'}")
    print(f"🌐 Translation: {'✅ Available' if TRANSLATION_AVAILABLE else '❌ Not Available'}")
    print(f"🤟 Sign Language: {'✅ Available' if SIGN_LANGUAGE_AVAILABLE else '❌ Not Available'}")
    print(f"🔊 Text-to-Speech: {'✅ Available' if TTS_AVAILABLE else '❌ Not Available'}")
    
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("\n📦 Install missing packages:")
        if not SPEECH_AVAILABLE:
            print("   pip install SpeechRecognition pyaudio")
        if not TRANSLATION_AVAILABLE:
            print("   pip install googletrans==4.0.0rc1")
        if not SIGN_LANGUAGE_AVAILABLE:
            print("   pip install mediapipe opencv-python")
        if not TTS_AVAILABLE:
            print("   pip install gtts")
        print()
    
    while True:
        print("\n📋 Select mode:")
        print(f"1. 🎤 Speech Translation {'✅' if SPEECH_AVAILABLE else '⌨️ Text Input'}")
        print(f"2. 🤟 Sign Language Translation {'✅' if SIGN_LANGUAGE_AVAILABLE else '❌'}")
        print("3. 🔄 Both Modes (Experimental)")
        print("4. ❌ Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            target_lang = get_language_code()
            if SPEECH_AVAILABLE:
                speech_translation_mode(target_lang=target_lang)
            else:
                print("🔄 Using text input mode instead of speech")
                text_input_mode(target_lang)
            
        elif choice == '2':
            if not SIGN_LANGUAGE_AVAILABLE:
                print("❌ Sign language recognition not available. Please install required packages.")
                continue
            sign_language_mode()
            
        elif choice == '3':
            print("🚧 Both modes simultaneously - Feature coming soon!")
            print("For now, please use modes separately.")
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Program terminated by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)