import tkinter as tk
import wave
from pyAudioAnalysis import audioSegmentation

def get_duration(audio_file):
    with wave.open(audio_file, 'rb') as wave_file:
        n_frames = wave_file.getnframes()
        frame_rate = wave_file.getframerate()
        duration = n_frames / float(frame_rate)
        return duration
def diarize_speakers(audio_file, n_speakers=2):
    try:
        [flagsInd, classesAll, acc, CM] = audioSegmentation.speaker_diarization(audio_file, n_speakers=n_speakers, plot_res=False)
        result_str = ""
        for i, seg in enumerate(classesAll):
            result_str += f"Speaker {seg}: {flagsInd[i, 0]} - {flagsInd[i, 1]}\n"
    except ValueError:
        result_str = "Speaker diarization unsuccessful. Please try again with a different audio file or adjust the parameters."
    return result_str

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.file_label = tk.Label(self, text="Audio file path:")
        self.file_label.pack()
        self.file_entry = tk.Entry(self)
        self.file_entry.pack()

        self.diarize_button = tk.Button(self, text="Diarize speakers", command=self.diarize)
        self.diarize_button.pack()

        self.result_label = tk.Label(self, text="Results:")
        self.result_label.pack()
        self.result_box = tk.Text(self, height=10)
        self.result_box.pack()

    def diarize(self):
        audio_file = self.file_entry.get()
        result_str = diarize_speakers(audio_file)
        self.result_box.delete('1.0', tk.END) # clear the result box
        self.result_box.insert(tk.END, result_str) # show the results in the result box

root = tk.Tk()
app = Application(master=root)
app.mainloop()
