import pyaudio
import threading
import time
import argparse
import wave
import tensorflow as tf
import torchaudio
import numpy as np
from ctcdecode import CTCBeamDecoder
from threading import Event

class Listener:
    def __init__(self, sample_rate=8000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nSpeech Recognition engine is now listening... \n")


class SpeechRecognitionEngine:
    def __init__(self, model_file, ken_lm_file, context_length=10):
        self.listener = Listener(sample_rate=8000)
        self.model = tf.keras.models.load_model("C:\Users\LENOVO\Desktop\speech_recognition\Models\05_sound_to_text\202405071029\model.h5")
        self.featurizer = torchaudio.transforms.MelSpectrogram(sample_rate=8000)
        self.audio_q = list()
        self.beam_decoder = CTCBeamDecoder(beam_width=100, model_path=ken_lm_file, alpha=0.75, beta=1.85)
        self.context_length = context_length * 50  # multiply by 50 because each 50 from output frame is 1 second
        self.out_args = None

    def save(self, waveforms, fname="audio_temp.wav"):
        wf = wave.open(fname, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(8000)
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return fname

    def predict(self, audio):
        fname = self.save(audio)
        waveform, _ = torchaudio.load(fname)
        log_mel = self.featurizer(waveform).unsqueeze(0).numpy()
        log_mel = np.expand_dims(log_mel, axis=-1)
        out = self.model.predict(log_mel)
        out = np.expand_dims(out, axis=0)
        decoded, _, _, out_lens = self.beam_decoder.decode(torch.Tensor(out))
        transcript = "".join([chr(x) for x in decoded[0][0][:out_lens[0][0]]])
        return transcript

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) < 5:
                continue
            else:
                pred_q = self.audio_q.copy()
                self.audio_q.clear()
                action(self.predict(pred_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                  args=(action,), daemon=True)
        thread.start()


class DemoAction:
    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""

    def __call__(self, x):
        results = x
        self.current_beam = results
        transcript = " ".join(self.asr_results.split() + results.split())
        print(transcript)
        self.asr_results = transcript


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the speech recognition engine in terminal.")
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to the trained TensorFlow model file.')
    parser.add_argument('--ken_lm_file', type=str, required=True,
                        help='Path to the KenLM binary file.')

    args = parser.parse_args()
    asr_engine = SpeechRecognitionEngine(args.model_file, args.ken_lm_file)
    action = DemoAction()

    asr_engine.run(action)
    threading.Event().wait()
