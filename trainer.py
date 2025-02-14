import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

#  1. KoGPT2 모델 및 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = GPT2TokenizerFast.from_pretrained(
    model_name,
    bos_token="</s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

#  2. Padding Token 설정 (GPT2는 기본적으로 padding token이 없음)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token  # 또는 '[PAD]' 사용 가능

#  3. 모델 로드 및 Token Embeddings 조정
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # 추가된 PAD 토큰 반영

#  4. M1 GPU (MPS) 활성화
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

#  5. 데이터셋 로드
dataset = load_dataset("text", data_files={"train": "gpt2_train_data.txt"})

#  6. 데이터 토크나이징 함수 (배치 처리)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

#  7. 데이터셋 변환 (토크나이징 적용)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 8. Train & Eval 데이터 분리 (20%를 검증 데이터로 사용)
split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 9. 데이터 Collator 설정 (Padding 적용)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 10. 훈련 설정 (MPS 최적화 적용)
training_args = TrainingArguments(
    output_dir="./kogpt2_results",
    num_train_epochs=200,  # 학습 Epochs
    per_device_train_batch_size=1,  # M1 메모리 최적화
    per_device_eval_batch_size=1,
    save_strategy="epoch",  # 매 Epoch 마다 저장
    evaluation_strategy="epoch",  # 매 Epoch 마다 평가
    logging_dir="./kogpt2_logs",
    logging_steps=50,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    save_total_limit=2,
    gradient_accumulation_steps=4,  # 작은 배치를 처리할 때 안정적인 학습 가능
    load_best_model_at_end=True,  # 가장 성능이 좋은 모델 저장
    metric_for_best_model="eval_loss",  # 손실 값이 가장 낮은 모델 선택
    greater_is_better=False,  # 손실 값이 낮을수록 좋은 모델
    report_to="none"
)

#  11. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 검증 데이터 추가
    data_collator=data_collator
)

# 12. 훈련 시작
trainer.train()

# 13. 훈련된 모델 저장
model.save_pretrained("./trained_kogpt2")
tokenizer.save_pretrained("./trained_kogpt2")
print(" KoGPT2 학습 완료 및 저장됨!")
