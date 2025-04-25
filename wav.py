import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import math

# Initialize MediaPipe Hands with higher confidence thresholds
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize Pygame audio mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Control parameters
frequency = 440  # Default frequency (A4 note)
speed_interval = 0.5  # Default interval between beeps (seconds)
volume = 0.5  # Default volume (0-1)
last_beep_time = 0
current_beep = None

# Smoothing parameters
SMOOTHING_FACTOR = 0.3
prev_frequency = frequency
prev_speed_interval = speed_interval
prev_volume = volume

# Musical note frequencies (A3 to A5)
NOTE_FREQUENCIES = {
    'A3': 220.00, 'B3': 246.94, 'C4': 261.63, 'D4': 293.66,
    'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00,
    'B4': 493.88, 'C5': 523.25, 'D5': 587.33, 'E5': 659.25,
    'F5': 698.46, 'G5': 783.99, 'A5': 880.00
}

def generate_beep(frequency, volume, duration=0.1):
    sample_rate = 44100
    samples = np.arange(int(duration * sample_rate))
    # Add a simple envelope to reduce clicks
    envelope = np.exp(-samples / (0.1 * sample_rate))
    wave = volume * envelope * np.sin(2 * np.pi * frequency * samples / sample_rate)
    stereo_wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(np.int16(stereo_wave * 32767))

def smooth_value(current, previous, factor=SMOOTHING_FACTOR):
    return previous + factor * (current - previous)

def find_closest_note(frequency):
    # Find the closest musical note to the given frequency
    closest_note = min(NOTE_FREQUENCIES.items(), 
                      key=lambda x: abs(x[1] - frequency))
    return closest_note[1]  # Return the frequency

def draw_control_info(frame, frequency, speed_interval, volume):
    # Create a semi-transparent overlay for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw control information with elegant styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Find and display the closest musical note
    closest_freq = find_closest_note(frequency)
    note_name = [k for k, v in NOTE_FREQUENCIES.items() if v == closest_freq][0]
    cv2.putText(frame, f"♪ {note_name} ({int(frequency)} Hz)", (10, 30),
                font, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"⚡ {speed_interval:.2f}s", (10, 60),
                font, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"🔊 {int(volume*100)}%", (10, 90),
                font, 0.8, (255, 255, 255), 2)
    
    # Draw minimal instructions
    cv2.putText(frame, "Left: Frequency | Right: Speed | Distance: Volume", (10, 710),
                font, 0.6, (255, 255, 255), 1)

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def draw_elegant_circle(frame, center, radius, color, thickness=2):
    # Draw outer glow
    for r in range(radius + 2, radius - 2, -1):
        alpha = (radius + 2 - r) / 4
        cv2.circle(frame, center, r, color, 1)
    # Draw main circle
    cv2.circle(frame, center, radius, color, thickness)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                           results.multi_handedness):
            hand_label = handedness.classification[0].label
            if hand_label == 'Left':
                left_hand = hand_landmarks.landmark
            elif hand_label == 'Right':
                right_hand = hand_landmarks.landmark

    # Process left hand (frequency control)
    if left_hand:
        index_tip = left_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = left_hand[mp_hands.HandLandmark.THUMB_TIP]
        
        # Draw elegant circles for index and thumb
        h, w, _ = frame.shape
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        
        draw_elegant_circle(frame, index_pos, 15, (0, 255, 0))
        draw_elegant_circle(frame, thumb_pos, 15, (0, 255, 0))
        
        # Draw subtle connecting line
        cv2.line(frame, index_pos, thumb_pos, (0, 255, 0), 1)
        
        dist = calculate_distance(index_tip, thumb_tip)
        raw_frequency = np.interp(dist, [0.05, 0.3], [220, 880])  # A3 to A5 range
        frequency = smooth_value(raw_frequency, prev_frequency)
        prev_frequency = frequency
    else:
        frequency = smooth_value(440, prev_frequency, 0.1)  # Return to A4 when no hand
        prev_frequency = frequency

    # Process right hand (speed control)
    if right_hand:
        index_tip = right_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = right_hand[mp_hands.HandLandmark.THUMB_TIP]
        
        # Draw elegant circles for index and thumb
        h, w, _ = frame.shape
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        
        draw_elegant_circle(frame, index_pos, 15, (255, 0, 0))
        draw_elegant_circle(frame, thumb_pos, 15, (255, 0, 0))
        
        # Draw subtle connecting line
        cv2.line(frame, index_pos, thumb_pos, (255, 0, 0), 1)
        
        dist = calculate_distance(index_tip, thumb_tip)
        raw_speed = np.interp(dist, [0.05, 0.3], [0.1, 1.0])  # More musical timing range
        speed_interval = smooth_value(raw_speed, prev_speed_interval)
        prev_speed_interval = speed_interval
    else:
        speed_interval = smooth_value(0.5, prev_speed_interval, 0.1)  # Return to default
        prev_speed_interval = speed_interval

    # Process distance between hands (volume control)
    if left_hand and right_hand:
        # Get left hand midpoint between index and thumb
        l_index = left_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        l_thumb = left_hand[mp_hands.HandLandmark.THUMB_TIP]
        l_mid_x = (l_index.x + l_thumb.x) / 2
        l_mid_y = (l_index.y + l_thumb.y) / 2
        
        # Get right hand midpoint between index and thumb
        r_index = right_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        r_thumb = right_hand[mp_hands.HandLandmark.THUMB_TIP]
        r_mid_x = (r_index.x + r_thumb.x) / 2
        r_mid_y = (r_index.y + r_thumb.y) / 2
        
        # Draw subtle line between hands
        h, w, _ = frame.shape
        left_pos = (int(l_mid_x * w), int(l_mid_y * h))
        right_pos = (int(r_mid_x * w), int(r_mid_y * h))
        cv2.line(frame, left_pos, right_pos, (255, 255, 255), 1)
        
        hands_dist = math.sqrt((l_mid_x - r_mid_x)**2 + (l_mid_y - r_mid_y)**2)
        raw_volume = np.interp(hands_dist, [0.1, 0.8], [0.0, 1.0])
        volume = smooth_value(raw_volume, prev_volume)
        prev_volume = volume
    else:
        volume = smooth_value(0.5, prev_volume, 0.1)  # Return to default
        prev_volume = volume

    # Play beeps based on current parameters
    current_time = time.time()
    if current_time - last_beep_time >= speed_interval:
        if current_beep:
            current_beep.stop()
        # Only generate and play sound if at least one hand is detected
        if left_hand or right_hand:
            # Use the closest musical note frequency
            note_frequency = find_closest_note(frequency)
            current_beep = generate_beep(note_frequency, volume)
            current_beep.play()
        else:
            current_beep = None
        last_beep_time = current_time

    # Draw control information
    draw_control_info(frame, frequency, speed_interval, volume)

    # Display frame
    cv2.imshow('Hand-Controlled Music Generator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
if current_beep:
    current_beep.stop()
cap.release()
cv2.destroyAllWindows()