# src/inference.py

from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from src.extract_text import extract_text_from_file


def load_model_and_tokenizer(model_path: str):
    """
    Load the fine-tuned QA model and tokenizer from the given path.
    Returns a HuggingFace pipeline for question-answering.
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline


def run_inference(file_path: str, qa_pipeline):
    """
    Run metadata extraction on the uploaded file using the QA model.
    Extracts text and runs inference for each metadata question.
    """
    # Extract text content from the document/image
    context = extract_text_from_file(file_path)

    # Normalize the context
    context = context.replace("\n", " ").replace(":", ": ").strip()

    # Define your metadata questions
    questions = [
        "What is the Agreement Value?",
        "What is the Agreement Start Date?",
        "What is the Agreement End Date?",
        "What is the Renewal Notice (Days)?",
        "Who is the Party One?",
        "Who is the Party Two?"
    ]

    predictions = {}
    for question in questions:
        try:
            result = qa_pipeline({"context": context, "question": question})
            answer = result.get("answer", "").strip()

            if not answer or answer.lower() in ["no answer", "none"]:
                predictions[question] = "Not Found"
            else:
                predictions[question] = answer
        except Exception as e:
            predictions[question] = f"Error: {str(e)}"

    return predictions
