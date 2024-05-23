import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
import matplotlib.pyplot as plt

# Turn on interactive mode
plt.ion()

class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, data: np.ndarray):
        if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join("temp", file.filename)
        try:

            file.save(file_path)
        
            spectrogram = get_spectrogram(wav_path)
            text = model.predict(spectrogram)
            return text
        
            os.remove(file_path)  # Clean up the saved file
        
            return jsonify({"prediction": text}), 200
        except Exception as e:
            error_message = str(e)
            error_trace = traceback.format_exc()
            return jsonify({"error": str(e)}), 500
        data_pred = np.expand_dims(data, axis=0)

        preds = self.model.run(self.output_names, {self.input_names[0]: data_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load(r"C:\Users\LENOVO\Desktop\speech_recognition\Models\05_sound_to_text\202405071029\configs.yaml")

    model = WavToTextModel(model_path=configs.model_path, char_list=configs.vocab, force_cpu=False)
    spectrogram = WavReader.get_spectrogram(wav_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
    padded_spectrogram = np.pad(spectrogram, ((0, configs.max_spectrogram_length - spectrogram.shape[0]),(0,0)), mode="constant", constant_values=0)
    text = model.predict(padded_spectrogram)
