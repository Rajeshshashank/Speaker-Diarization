import wave
import os
from pyAudioAnalysis import audioSegmentation
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def get_duration(audio_file):
    with wave.open(audio_file, 'rb') as wave_file:
        n_frames = wave_file.getnframes()
        frame_rate = wave_file.getframerate()
        duration = n_frames / float(frame_rate)
        return duration


def diarize_speakers(audio_file, n_speakers=1):
    try:
        [flagsInd, classesAll, acc, CM] = audioSegmentation.speaker_diarization(audio_file, n_speakers=n_speakers,
                                                                                plot_res=False)
        result_str = ""
        for i, seg in enumerate(classesAll):
            result_str += f"Speaker {seg}: {flagsInd[i, 0]} - {flagsInd[i, 1]}\n"
    except ValueError:
        result_str = "Speaker diarization unsuccessful. Please try again with a different audio file or adjust the parameters."
    return result_str


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio']

    # save the audio file
    filename = audio_file.filename
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(audio_path)

    # perform speaker diarization
    results = diarize_speakers(audio_path)

    # format the results as a string
    result_str = ""
    for i, speaker in enumerate(results):
        result_str += f"Speaker {i + 1}: {speaker}\n"

    # return the results as a JSON object
    return jsonify({'results': result_str})


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = './uploads'
    app.run(debug=True)
