from pyAudioAnalysis import audioSegmentation

def diarize_speakers(audio_file):
    segments = audioSegmentation.speaker_diarization(audio_file)
    result_str = ""
    for segment in segments:
        result_str += 'Speaker %d: %d - %d\n' % (segment['speaker'], segment['start'], segment['end'])
    return result_str
