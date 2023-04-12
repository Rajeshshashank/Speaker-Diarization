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
