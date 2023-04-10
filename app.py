import numpy as np
import librosa
import webrtcvad
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks
from scipy.io import wavfile
from sklearn.cluster import KMeans
from flask import Flask, render_template, request, jsonify

from diarization import diarize_speakers

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_data = request.files['audio'].read()

    # convert audio data to numpy array
    audio_np = librosa.load(audio_data, sr=16000)[0]

    # perform speaker diarization
    results = diarize_speakers(audio_np)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
