from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_emotion import predict_emotion_from_file
import os, traceback

app = Flask(__name__)
CORS(app)

UPLOADS_FOLDER = "uploads"
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    # 1) Controllo che ci sia davvero un file nel form
    if "audio" not in request.files:
        return jsonify({'error': 'No audio file in request'}), 400

    audio_file = request.files["audio"]
    file_path = os.path.join(UPLOADS_FOLDER, audio_file.filename)
    audio_file.save(file_path)
    print(f"[server] Saved upload to {file_path}")

    try:
        # 2) Invoco la tua routine, che a sua volta fa tutti i print interni
        emotion = predict_emotion_from_file(file_path)
        print(f"[server] Predizione restituita: {emotion}")
        return jsonify({'emotion': emotion}), 200

    except Exception as e:
        # 3) Se entra qui, stampo lo stack completo sul server e rispondo 500
        print("[server] ERRORE in /predict_emotion:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Attento a metterlo in host 0.0.0.0 e porta corretta
    app.run(host='0.0.0.0', port=5000)
