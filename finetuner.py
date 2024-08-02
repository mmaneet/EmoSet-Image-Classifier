import evaluate
from torch.utils.data import DataLoader
from torchvision.datasets.EmoSet2 import EmoSet2
from datasets import Dataset as HFDataset, DatasetDict, Features, ClassLabel, Image

from transformers import logging, AutoModelForImageClassification, TrainingArguments, Trainer

model_checkpoint = "google/vit-base-patch16-224"  # pre-trained model from which to fine-tune
batch_size = 32  # batch size for training and evaluation

data_root = r"C:\Users\manee\EmoSet-118K-7"
num_emotion_classes = 8
phase = 'val'
print(data_root)

t_dataset = EmoSet2(
    data_root=data_root,
    num_emotion_classes=num_emotion_classes,
    phase='train',
)

v_dataset = EmoSet2(
    data_root=data_root,
    num_emotion_classes=num_emotion_classes,
    phase='val',
)

label2idx = {
    "amusement": 0,
    "awe": 1,
    "contentment": 2,
    "excitement": 3,
    "anger": 4,
    "disgust": 5,
    "fear": 6,
    "sadness": 7
}

idx2label =  {
    "0": "amusement",
    "1": "awe",
    "2": "contentment",
    "3": "excitement",
    "4": "anger",
    "5": "disgust",
    "6": "fear",
    "7": "sadness"
}

print("asdkaljf")
train_dict = [t_dataset[i] for i in range(len(t_dataset))]
val_dict = [v_dataset[i] for i in range(len(v_dataset))]

print("kalsdjalks")
train_ds = HFDataset.from_list(train_dict)
val_ds = HFDataset.from_list(val_dict)

print("talkjasf ??")


print(train_ds)
print(val_ds)
# print(val_ds[0])

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id = label2idx,
    id2label = idx2label,
    ignore_mismatched_sizes = True,
)

# Freeze all parameters except the classification head
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classification head parameters
for param in model.classifier.parameters():
    param.requires_grad = True

model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-emoset",
    remove_unused_columns=False,
    evaluation_strategy = "steps",
    save_strategy = "steps",
    eval_steps=30,
    save_steps=30,
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=20,
    logging_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

import numpy as np

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=eval_pred.label_ids)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")["precision"],
        "recall": recall.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")["recall"],
    }

import torch

def collate_fn(examples):
    pixel_values = torch.stack([torch.tensor(example["pixel_value"]) for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()