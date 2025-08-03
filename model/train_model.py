from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

MODEL_NAME = "deepset/roberta-base-squad2"

print("🔄 Loading dataset...")
dataset = load_dataset("json", data_files={"train": "dataset/train_dataset.json"})
raw_data = dataset["train"][0]["data"]

def flatten_squad(example):
    flattened = []
    for paragraph in example["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            answers = qa.get("answers", [])
            if answers:
                flattened.append({
                    "context": context,
                    "question": qa["question"],
                    "answers": answers[0]  # one answer per question
                })
    return flattened

flattened_data = []
for item in raw_data:
    flattened_data.extend(flatten_squad(item))

train_dataset = Dataset.from_list(flattened_data)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    question = example["question"]
    context = example["context"]

    # 🛠 Get the first answer only (can be extended for multiple)
    answer = example["answers"]
    if isinstance(answer, list):
        answer = answer[0]

    start_char = answer["answer_start"]
    end_char = start_char + len(answer["text"])
    
    # Tokenize
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = inputs.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = example["answers"]
        if isinstance(answers, list):
            answer = answers[0]
        else:
            answer = answers

        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        # Find start and end token positions
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (start_char >= offsets[token_start_index][0] and end_char <= offsets[token_end_index][1]):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Otherwise find start and end token positions
            for idx in range(token_start_index, token_end_index + 1):
                if offsets[idx][0] <= start_char and offsets[idx][1] > start_char:
                    start_positions.append(idx)
                if offsets[idx][0] < end_char and offsets[idx][1] >= end_char:
                    end_positions.append(idx)
                    break

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


tokenized_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)

model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

args = TrainingArguments(
    output_dir="model_output",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
