import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_path = "C:/Users/pradu/OneDrive/Desktop/metadata_extraction_project/model_output/checkpoint-93"
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

with open("dataset/test_dataset.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

correct = 0
total = 0
missing = 0

for doc in test_data["data"]:
    for para in doc["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            if qa.get("is_impossible", False) or not qa["answers"]:
                missing += 1
                continue

            question = qa["question"]
            true_answer = qa["answers"][0]["text"].strip().lower()

            try:
                result = qa_pipeline(question=question, context=context)
                pred_answer = result["answer"].strip().lower()
            except Exception as e:
                print(f"❌ Error: {e}")
                pred_answer = ""

            print(f"Q: {question}")
            print(f"✅ Expected: {true_answer}")
            print(f"🤖 Predicted: {pred_answer}\n")

            if true_answer in pred_answer or pred_answer in true_answer:
                correct += 1
            total += 1

print("📊 Evaluation Results")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Missing: {missing}")
print(f"Recall: {correct / total:.2f}" if total else "Recall: N/A")
