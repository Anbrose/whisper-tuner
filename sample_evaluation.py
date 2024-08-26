import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import librosa

# 读取wav文件
data, sampling_rate = librosa.load("/data/nhi-dictation-dataset-wav/audio/cb077b4c-328c-11ef-9191-0283cfceb214.wav")
audio_data = {
    'path': '/data/nhi-dictation-dataset-wav/audio/cb077b4c-328c-11ef-9191-0283cfceb214.wav',
    'array': data,
    'sampling_rate': sampling_rate
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "/whisper-tuner/models/whisper-large-v3-finetuned/checkpoint-111"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe(audio_data)
print(result["text"])
