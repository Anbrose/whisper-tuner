import torch
import evaluate
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, load_from_disk, Dataset
import librosa
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
metric = evaluate.load("wer")

model_id = "/whisper-tuner/models/whisper-large-v3-finetuned-30s/checkpoint-269"

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

dataset = load_from_disk("/data/gpt4o-cleansed-nhi-wav-30s-test")

data_list = []
for index, d in enumerate(tqdm(dataset)):
    file_path = d['audio'].replace(".webm", ".wav")
    data, sampling_rate = librosa.load(file_path)
    audio_data = {
        'path': file_path,
        'array': data,
        'sampling_rate': sampling_rate
    }
    result = pipe(audio_data)
    d['tuned_result'] = result["text"]
    metric.add_batch(predictions=[result["text"]], references=[d['transcription']])
    print("Current", metric.compute())
    print("*********Result********")
    print("Tuned: ", result["text"])
    print("*********Original********")
    print("Original: ", d['transcription'])

    data_list.append(d)
    break


Dataset.from_list(data_list).save_to_disk("dataset/result/whisper-tuned-nhi-dataset")
