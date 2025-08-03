#  Metadata Extraction AI

> This project extracts structured metadata (like Agreement Dates, Parties, Value) from `.docx` and `.png` contract files using a fine-tuned Question Answering (QA) model.

---
## Project description 
This project is an AI-based metadata extraction system that automatically extracts key information (like Agreement Value, Start Date, End Date, Parties, etc.) from contract documents in .docx or .png formats. It uses a fine-tuned transformer-based Question Answering (QA) model from HuggingFace and provides a FastAPI-based REST API to upload documents and get structured metadata in response.

The system includes:

Text extraction from files

QA-style dataset preparation from structured CSV

Model training using HuggingFace Transformers

Field-wise prediction and evaluation

RestAPI
for better outputand fast training use gpu 

---
##  Project Structure

```
metadata_extraction_project/
├── api/               ← FastAPI app
├── data/              ← Training/testing files & CSV
├── dataset/           ← QA-style dataset scripts
├── model/             ← Training & evaluation 
├── model_output/      ← trained model output
scripts
├── src/               ← Inference and extracttext extraction
├── README.md
├── requirements.txt
```

---

##  Setup Instructions

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

##  Train the Model

```bash
python dataset/prepare_dataset.py
python model/train_model.py

by runnig this it will generate train model in model_ouput

```

---

##  Evaluate the Model

```bash
python model/evaluate.py
```

---

##  Run the API

```bash
python api/app.py
```

Visit Swagger UI at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

##  API Endpoint

**POST** `/extract/`  
Upload a `.docx` or `.png` file to extract metadata.

**Returns JSON** like:

```json
{
  "predictions": {
    "What is the Agreement Value?": "12,000",
    "What is the Agreement Start Date?": "15th December 2012 ",
    "What is the Agreement End Date?": "14th November 2013",
    "What is the Renewal Notice (Days)?": "60",
    "Who is the Party One?": "V.K. NATARAJ, son of V. KANDASWAMI CHETTIAR, aged 55 years, residing at Door No - 5/8, TYPE, 4th MAIN ROAD, SIDCO NAGAR, VILLIVAKKAM, CHENNAI-600049",
    "Who is the Party Two?": "s.sakunthala"
  }
}
```

---

##  Technologies Used

- Python, FastAPI, Uvicorn
- HuggingFace Transformers (QA model)
- PDF/DOCX/PNG text extraction
- Model evaluation with per-field recall

---

##  Notes

- Ensure model is trained and saved in `model_output/` before running the API.
- For inference, model should support SQuAD-style QA.

---

## git repo for assigment2(loddtype classification) two is in below link
https://github.com/pradumnyaastage/Loadtype_classification.git
