from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from functools import lru_cache

app = Flask(__name__)

# 1. Configurazione MediaPipe IDENTICA allo script Python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Modalità video (come nello script)
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Lista classi ESATTAMENTE come nello script Python
GESTURE_NAMES = [
    "0", "1", "2", "3", "4",
    "5", "pistola", "letsgosky", "ok"
]


# 3. Funzione di normalizzazione IDENTICA
def normalize_hand(hand):
    hand = np.array(hand).reshape(-1, 3)
    base = hand[0]  # Polso (indice 0)
    mid = hand[9]  # Base del medio (indice 9)
    distanza = np.linalg.norm(base - mid)
    hand_normalized = (hand - base) / distanza
    return hand_normalized.flatten().tolist()


# 4. Caricamento modello con cache
@lru_cache(maxsize=1)
def load_model():
    model = tf.keras.models.load_model('Phand3.1.h5')
    print(f"Modello caricato. Input shape atteso: {model.input_shape}")
    return model


# 5. Route principale
@app.route('/')
def home():
    return render_template('index.html')


# 6. Route per predizioni (con normalizzazione)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 6.1 Leggi immagine
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 6.2 Rileva landmarks
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return jsonify({"error": "Nessuna mano rilevata"}), 400

        # 6.3 Estrai e normalizza
        hand_landmarks = results.multi_hand_landmarks[0]
        raw_landmarks = []
        for lm in hand_landmarks.landmark:
            raw_landmarks.extend([lm.x, lm.y, lm.z])

        normalized_landmarks = normalize_hand(raw_landmarks)

        # 6.4 Debug: confronta con lo script Python
        print("\n--- DEBUG ---")
        print("Primi 5 valori normalizzati:", normalized_landmarks[:5])

        # 6.5 Predizione
        model = load_model()
        input_data = np.array(normalized_landmarks).reshape(1, -1).astype(np.float32)
        prediction = model.predict(input_data, verbose=0)

        # 6.6 Risposta
        gesture_id = np.argmax(prediction)
        return jsonify({
            "gesture_id": int(gesture_id),
            "gesture_name": GESTURE_NAMES[gesture_id],
            "confidence": float(prediction[0][gesture_id])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)