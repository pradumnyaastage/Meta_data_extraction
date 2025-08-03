import os
import pandas as pd
import json
from src.extract_text import extract_text

csv_path = "data/train.csv"
docs_folder = "data/train"
output_path = "dataset/train_dataset.json"

COLUMN_MAP = {
    "Aggrement Value": "Agreement Value",
    "Aggrement Start Date": "Agreement Start Date",
    "Aggrement End Date": "Agreement End Date",
    "Renewal Notice (Days)": "Renewal Notice (Days)",
    "Party One": "Party One",
    "Party Two": "Party Two"
}

df = pd.read_csv(csv_path)
df.rename(columns=COLUMN_MAP, inplace=True)

data = {"data": []}

for idx, row in df.iterrows():
    file_name = row['File Name'].strip()
    matched_path = None

    for ext in ['.docx', '.png']:
        path = os.path.join(docs_folder, file_name + ext)
        if os.path.exists(path):
            matched_path = path
            break

    if not matched_path:
        print(f"❌ File not found: {file_name}")
        continue

    try:
        context = extract_text(matched_path)
    except Exception as e:
        print(f"❌ Extract error for {file_name}: {e}")
        continue

    qas = []
    for field in ["Agreement Value", "Agreement Start Date", "Agreement End Date",
                  "Renewal Notice (Days)", "Party One", "Party Two"]:
        answer = str(row[field]).strip()
        answer_start = context.find(answer)
        if answer and answer_start != -1:
            qas.append({
                "question": f"What is the {field}?",
                "id": f"{idx}_{field.replace(' ', '_')}",
                "answers": [{"text": answer, "answer_start": answer_start}],
                "is_impossible": False
            })
        else:
            qas.append({
                "question": f"What is the {field}?",
                "id": f"{idx}_{field.replace(' ', '_')}",
                "answers": [],
                "is_impossible": True
            })

    data["data"].append({
        "title": file_name,
        "paragraphs": [{
            "context": context,
            "qas": qas
        }]
    })

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("✅ Dataset prepared and saved to:", output_path)
