import os
import pandas as pd
import json
from src.extract_text import extract_text

csv_path = "data/test.csv"
docs_folder = "data/test"
output_path = "dataset/test_dataset.json"

# Standardize column names
COLUMN_MAP = {
    "Aggrement Value": "Agreement Value",
    "Aggrement Start Date": "Agreement Start Date",
    "Aggrement End Date": "Agreement End Date",
    "Renewal Notice (Days)": "Renewal Notice (Days)",
    "Party One": "Party One",
    "Party Two": "Party Two"
}

print("🔁 Reading CSV...")
df = pd.read_csv(csv_path)

print("🔁 Renaming columns...")
df.rename(columns=COLUMN_MAP, inplace=True)

print("📄 Generating QA-style test dataset...")
data = {"data": []}

for idx, row in df.iterrows():
    file_name = row['File Name'].strip()

    # Match file with extension
    possible_extensions = ['.docx', '.png']
    matched_path = None
    for ext in possible_extensions:
        candidate = os.path.join(docs_folder, file_name + ext)
        if os.path.exists(candidate):
            matched_path = candidate
            break

    if not matched_path:
        print(f"❌ File not found with known extensions for: {file_name}")
        continue

    file_path = matched_path

    try:
        context = extract_text(file_path)
    except Exception as e:
        print(f"❌ Failed to extract text from {file_path}: {e}")
        continue

    qas = []
    for field in ['Agreement Value', 'Agreement Start Date', 'Agreement End Date',
                  'Renewal Notice (Days)', 'Party One', 'Party Two']:
        answer = str(row[field]).strip()
        answer_start = context.find(answer)

        if answer_start == -1:
            qas.append({
                "question": f"What is the {field}?",
                "id": f"{idx}_{field.replace(' ', '_')}",
                "answers": [],
                "is_impossible": True
            })
        else:
            qas.append({
                "question": f"What is the {field}?",
                "id": f"{idx}_{field.replace(' ', '_')}",
                "answers": [{"text": answer, "answer_start": answer_start}],
                "is_impossible": False
            })

    data["data"].append({
        "title": file_name,
        "paragraphs": [{
            "context": context,
            "qas": qas
        }]
    })

print(f"💾 Saving to {output_path}...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("✅ Test dataset preparation complete.")
