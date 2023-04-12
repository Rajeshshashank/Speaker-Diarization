import io
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
    # extract the uploaded file from the request object
    file = request.files['audio']

    # convert the file data to a format that can be read by librosa
    audio_data = AudioSegment.from_file(file)
    audio_data = audio_data.set_frame_rate(16000)
    audio_data = audio_data.set_channels(1)
    audio_data.export('temp.wav', format='wav')
    rate, wav_data = wavfile.read('temp.wav')

    # perform speaker diarization
    results = diarize_speakers(wav_data)

    # format the results as a string
    result_str = ""
    for i, speaker in enumerate(results):
        result_str += f"Speaker {i + 1}: {speaker}\n"

    # return the results as a JSON object
    return jsonify({'results': result_str})


if __name__ == '__main__':
    app.run(debug=True)
