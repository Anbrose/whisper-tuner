import pandas as pd
import datasets
import soundfile as sfpip
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from pydub import AudioSegment
from datasets import Audio, load_from_disk
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "/whisper-tuner/models/whisper-large-finetuned/checkpoint-111", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def preprocess_function(examples):
    inputs = processor(
        examples["audio"]["array"],
        sampling_rate=examples["audio"]["sampling_rate"],
        return_tensors="pt"
    )

    labels = processor.tokenizer(examples["transcription"], return_tensors="pt").input_ids

    return {"input_features": inputs.input_features.squeeze(), "labels": labels.squeeze()}


dataset = load_from_disk("/data/gpt4o-cleansed-nhi-wav")

dataset = dataset.cast_column(
    "audio",
    Audio(
        sampling_rate=16000,
    ),
)

# remove those label longer than 2000
processed_dataset = dataset.map(preprocess_function, remove_columns=["audio_file", "transcription", "audio"])

processed_dataset.set_format(type="torch", columns=["input_features", "labels"])

for ds in dataset:
    result = pipe(ds["audio"])
    print(result["text"])
    break
