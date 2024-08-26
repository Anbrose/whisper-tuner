import pandas as pd
import datasets
import soundfile as sfpip
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import Audio, load_from_disk


# 加载模型和处理器
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")


def preprocess_function(examples):
    inputs = processor(
        examples["audio"]["array"],
        sampling_rate=examples["audio"]["sampling_rate"],
        return_tensors="pt"
    )
    with processor.as_target_processor():
        labels = processor.tokenizer(examples["transcription"], return_tensors="pt").input_ids

    return {"input_features": inputs.input_features.squeeze(), "labels": labels.squeeze()}


dataset = load_from_disk("/data/gpt4o-cleansed-nhi-wav")

dataset = dataset.cast_column(
    "audio",
    Audio(
        sampling_rate=16000,
    ),
)
processed_dataset = dataset.map(preprocess_function, remove_columns=["audio_file", "transcription", "audio"])

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-finetuned",
    per_device_train_batch_size=4,
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
