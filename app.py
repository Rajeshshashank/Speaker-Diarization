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
    audio_file = request.files['audio']

    # convert audio file to WAV format
    audio_data = AudioSegment.from_file(audio_file)
    audio_wav = audio_data.export(format='wav')

    # perform speaker diarization
    results = diarize_speakers(audio_wav)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
