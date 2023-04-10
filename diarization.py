from pyAudioAnalysis import audioSegmentation

def diarize_speakers(audio_file):
    segments = audioSegmentation.speaker_diarization(audio_file)
    for segment in segments:
        print('Speaker %d: %d - %d' % (segment['speaker'], segment['start'], segment['end']))
