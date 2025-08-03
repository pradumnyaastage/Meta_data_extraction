# api/app.py

import os
import sys
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile

sys.path.append(os.path.abspath("."))

from src.inference import load_model_and_tokenizer, run_inference

app = FastAPI(title="Metadata Extraction API")

# Load fine-tuned QA model
qa_pipeline = load_model_and_tokenizer("C:/Users/pradu/OneDrive/Desktop/metadata_extraction_project/model_output/checkpoint-93")

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/extract/")
async def extract_metadata(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        predictions = run_inference(tmp_path, qa_pipeline)
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    import threading
    import webbrowser

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8000/docs")

    threading.Timer(1.5, open_browser).start()
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
