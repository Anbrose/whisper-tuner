import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, load_from_disk, Dataset
import librosa
from tqdm import tqdm

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

dataset = load_from_disk("/data/gpt4o-cleansed-nhi-wav")

data_list = []
for d in tqdm(dataset):
    file_path = d['audio_file']
    data, sampling_rate = librosa.load("/data/nhi-dictation-dataset-wav/audio/{}".format(file_path))
    audio_data = {
        'path': '/data/nhi-dictation-dataset-wav/audio/{}'.format(file_path),
        'array': data,
        'sampling_rate': sampling_rate
    }
    result = pipe(audio_data)
    d['tuned_result'] = result["text"]
    data_list.append(d)

Dataset.from_list(data_list).save_to_disk("dataset/result/whisper-tuned-nhi-dataset")
