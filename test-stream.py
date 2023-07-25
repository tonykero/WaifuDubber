import sounddevice as sd
import torch
import torch_directml
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import time

class Model:
    def __init__(self, model_id, source_lang, gpu=False):
        self._model_id = model_id
        self.gpu = gpu
        
        self.device = torch_directml.device()
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        
        if self.gpu:
            self.model = self.model.to(self.device)

        self.decoder = self.processor.get_decoder_prompt_ids(language=source_lang, task='translate')

    def transcribe(self, sample, samplerate):
        with torch.no_grad():
            proc_outputs = self.processor(sample, sampling_rate=samplerate,return_tensors="pt")
            features = proc_outputs.input_features
            inputs = features
            if self.gpu:
                inputs = inputs.to(self.device)

            predicted_ids = self.model.generate(inputs, forced_decoder_ids=self.decoder)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return transcription


model = Model("openai/whisper-medium","japanese", gpu=True)
#model = Model("bofenghuang/whisper-large-v2-french", "french", gpu=True)
stereo_mix_device = 2
num_channels = 1
samplerate= 16000
blocksize=5
def callback(data, frame_count, time_info, status):
    start = time.time()
    transcription = model.transcribe(data.flatten(), samplerate)
    end = time.time()

    if len(transcription) > 0:
        transcription = transcription[0]
    print(transcription + " (computed in {:.2f} seconds)".format(end-start))
    

with sd.InputStream(device=stereo_mix_device, channels=num_channels, callback=callback, blocksize=int(samplerate * 5),samplerate=samplerate):
    print("Stream ready")
    while True:
        response = input()
        if response == 'q':
            break