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
import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")


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
processed_dataset = dataset.map(preprocess_function, remove_columns=["audio_file", "transcription", "audio"]).map(
    lambda x: x if x["labels"].shape[0] < 200 else None
)


processed_dataset.set_format(type="torch", columns=["input_features", "labels"])
for ds in processed_dataset:
    print(ds["input_features"].shape, ds["labels"].shape)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    num_train_epochs=1,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    warmup_steps=500,
    save_total_limit=3,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()
