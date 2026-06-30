from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from gtts import gTTS
import webbrowser
import tkinter as tk
import threading
import atexit

root = tk.Tk()
root.withdraw()

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)
cap.set(cv2.CAP_PROP_EXPOSURE, -3)

SUPER_CONCENTRATION_THRESHOLD = 0.04
CONCENTRATION_THRESHOLD = 0.02
DROWSINESS_THRESHOLD = 0.03
DROWSINESS_ALERT_TIME = 3
DROWSINESS_GAME_TIME = 30
MIN_FACE_AREA = 20000
MAX_FACE_AREA = 150000
ZONE_OUT_FACE_AREA = 40000
BREAK_DURATION = 60

drowsy_start = None
zone_out_start = None
alert_triggered = False
thumbs_up_triggered = False
voice_prompt_triggered = False
prev_time = time.time()
focus_time = 0.0
flash_counter = 0

def wake_up_alert():
    global alert_triggered
    if not alert_triggered:
        def play_alert():
            try:
                tts = gTTS("Wake up! You seem tired! Stay focused!", lang="en")
                tts.save("alert.mp3")
                pygame.mixer.init()
                pygame.mixer.music.load("alert.mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.5)
            except Exception as e:
                print("Error in wake_up_alert:", e)
        threading.Thread(target=play_alert).start()
        alert_triggered = True

def flash_screen():
    global flash_counter
    flash_counter = 6

def prompt_memory_game_voice():
    try:
        tts = gTTS("You seem tired, would you like to take a one minute break and play a memory game? If yes, please show a thumbs up!", lang="en")
        tts.save("prompt.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("prompt.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.5)
    except Exception as e:
        print("Error in prompt_memory_game_voice:", e)

def eye_vertical_opening(upper, lower):
    return abs(upper.y - lower.y)

def is_thumbs_up(hand_landmarks):
    tolerance = 0.05  # Adjust tolerance as needed

    # Retrieve landmarks for the thumb
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y

    # Retrieve landmarks for the index finger
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

    # Retrieve landmarks for the middle, ring, and pinky fingers
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

    # For a thumbs-up gesture, we expect:
    # - The thumb to be extended upward:
    #   (i.e. thumb_tip is higher than thumb_ip and thumb_ip is higher than thumb_mcp)
    thumb_extended = (thumb_tip < thumb_ip - tolerance) and (thumb_ip < thumb_mcp - tolerance)

    # - The other fingers to be folded:
    #   (i.e. their tip positions are lower than their MCP joints)
    index_folded = index_tip > index_mcp + tolerance
    middle_folded = middle_tip > middle_mcp + tolerance
    ring_folded = ring_tip > ring_mcp + tolerance
    pinky_folded = pinky_tip > pinky_mcp + tolerance

    if thumb_extended and index_folded and middle_folded and ring_folded and pinky_folded:
        return True
    return False


def reset_thumbs_up_flag():
    global thumbs_up_triggered
    thumbs_up_triggered = False

def gen_frames():
    global drowsy_start, zone_out_start, alert_triggered, prev_time, focus_time, thumbs_up_triggered, flash_counter, voice_prompt_triggered
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame captured")
            break
        print("Average pixel value:", frame.mean())
        h, w, _ = frame.shape
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = face_mesh.process(frame_rgb)
        except ValueError as e:
            print("Face mesh error:", e)
            continue
        state = "CONCENTRATING"
        box_color = (0, 255, 0)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)
                face_area = (x_max - x_min) * (y_max - y_min)
                if face_area < MIN_FACE_AREA:
                    state = "FACE TOO SMALL"
                    box_color = (255, 255, 0)
                    drowsy_start = None
                    alert_triggered = False
                    zone_out_start = None
                elif face_area > MAX_FACE_AREA:
                    state = "FACE TOO CLOSE"
                    box_color = (255, 0, 255)
                    drowsy_start = None
                    alert_triggered = False
                    zone_out_start = None
                elif face_area < ZONE_OUT_FACE_AREA:
                    if zone_out_start is None:
                        zone_out_start = time.time()
                        state = "CONCENTRATING"
                        box_color = (0, 255, 0)
                    else:
                        zone_elapsed = time.time() - zone_out_start
                        if zone_elapsed >= 5:
                            state = "SEEMS ZONED OUT"
                            box_color = (0, 165, 255)
                        else:
                            state = "CONCENTRATING"
                            box_color = (0, 255, 0)
                    drowsy_start = None
                    alert_triggered = False
                else:
                    zone_out_start = None
                    left_upper = face_landmarks.landmark[159]
                    left_lower = face_landmarks.landmark[145]
                    right_upper = face_landmarks.landmark[386]
                    right_lower = face_landmarks.landmark[374]
                    left_opening = eye_vertical_opening(left_upper, left_lower)
                    right_opening = eye_vertical_opening(right_upper, right_lower)
                    avg_opening = (left_opening + right_opening) / 2.0
                    cv2.putText(frame, f"Eye Open: {avg_opening:.3f}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if avg_opening >= SUPER_CONCENTRATION_THRESHOLD:
                        state = "SUPER CONCENTRATING"
                        box_color = (0, 255, 255)
                        drowsy_start = None
                        alert_triggered = False
                        focus_time += delta_time
                    elif avg_opening >= CONCENTRATION_THRESHOLD:
                        state = "CONCENTRATING"
                        box_color = (0, 255, 0)
                        drowsy_start = None
                        alert_triggered = False
                        focus_time += delta_time
                    elif avg_opening >= DROWSINESS_THRESHOLD:
                        state = "SEEMS ZONED OUT"
                        box_color = (0, 165, 255)
                        drowsy_start = None
                        alert_triggered = False
                    else:
                        state = "DROWSY"
                        box_color = (0, 0, 255)
                        if drowsy_start is None:
                            drowsy_start = time.time()
                            alert_triggered = False
                            voice_prompt_triggered = False
                        else:
                            elapsed = time.time() - drowsy_start
                            cv2.putText(frame, f"Drowsy for {elapsed:.1f}s", (30, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            if elapsed >= DROWSINESS_ALERT_TIME and elapsed < DROWSINESS_GAME_TIME and not alert_triggered:
                                wake_up_alert()
                                flash_screen()
                                alert_triggered = True
                            if elapsed >= DROWSINESS_GAME_TIME:
                                cv2.putText(frame, "Break time: Would you like to play a memory game?", (30, 110),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.putText(frame, "If yes, please show a thumbs up!", (30, 150),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                if not voice_prompt_triggered:
                                    prompt_memory_game_voice()
                                    voice_prompt_triggered = True
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.putText(frame, state, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
        try:
            hand_results = hands.process(frame_rgb)
        except Exception as e:
            print("Hand detection error:", e)
            hand_results = None
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if is_thumbs_up(hand_landmarks):
                    cv2.putText(frame, "Thumbs Up Detected!", (30, h - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if not thumbs_up_triggered:
                        print("Thumbs up detected! Redirecting to memory game website.")
                        webbrowser.open("https://www.memozor.com/memory-games-for-adults")
                        thumbs_up_triggered = True
                        threading.Timer(10, reset_thumbs_up_flag).start()
        if flash_counter > 0:
            if flash_counter % 2 == 0:
                frame = np.ones_like(frame) * 255
            else:
                frame = np.zeros_like(frame)
            flash_counter -= 1
        cv2.putText(frame, f"Focus Time: {focus_time/60.0:.2f} min", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@atexit.register
def cleanup():
    if cap.isOpened():
        cap.release()

if __name__ == '__main__':
    app.run(debug=True)
