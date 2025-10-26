# gemini_vlm_grader.py
"""
Gemini-based intelligent exam evaluator:
- Extracts student details from 1st page (name, reg no, course code)
- Reads answer sheet (handwritten/typed)
- Uses question paper (with or without embedded answer key)
- Awards marks and justifies scoring per question
- Detects conceptual weaknesses and common mistakes
"""

import os
import json
import time
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types, errors
import logging

logger = logging.getLogger(__name__)

# ---------------------------
# CONFIG
# ---------------------------
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY environment variable before running.")

PRIMARY_MODEL = "gemini-2.5-pro"
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview",
    "gemini-2.0-flash",
    "gemma-3",
    "gemini-2.5-flash-lite",
]

OUT_DIR = "graded_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# helper: retry/backoff
# ---------------------------
def retry_backoff(fn, attempts: int = 6, base_sleep: float = 1.0):
    ex = None
    for i in range(attempts):
        try:
            return fn()
        except errors.APIError as e:
            ex = e
            code = getattr(e, "code", None)
            msg = getattr(e, "message", str(e))
            if code == 429 or "rate" in msg.lower() or 500 <= (code or 0) < 600:
                sleep = base_sleep * (2 ** i) + base_sleep * (0.25 * i)
                logger.info(f"[retry] attempt {i+1} -> sleeping {sleep:.1f}s due to: {msg}")
                time.sleep(sleep)
                continue
            raise
    raise ex or RuntimeError("Retries exhausted")

# ---------------------------
# core functions
# ---------------------------
def create_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)

def upload_file(client: genai.Client, filepath: str) -> Any:
    logger.info(f"Uploading file {filepath} ...")
    fobj = client.files.upload(file=filepath)
    logger.info("Uploaded, got file object:", getattr(fobj, "name", "unnamed"))
    return fobj

# 🧠 --- UPDATED PROMPT FUNCTION ---
def build_grading_prompt() -> str:
    return (
        "You are an impartial and consistent exam evaluator. "
        "You are given TWO files:\n"
        "1️⃣ Student's handwritten or typed answer sheet (multi-page PDF)\n"
        "2️⃣ A question paper file that may contain answer keys with the questions, "
        "or only the questions.\n\n"
        "Your tasks:\n"
        "- From the *first page* of the student's sheet, extract:\n"
        "  • student_name\n  • registration_number\n  • course_code (if available)\n"
        "- For each question, identify the student's answer and evaluate it conceptually "
        "against the provided question paper or embedded answer key.\n"
        "- If an explicit key exists, use it as reference.\n"
        "- If only questions exist, infer correctness intelligently using your subject knowledge.\n"
        "- Award marks fairly: full marks for conceptually correct answers, partial marks "
        "for incomplete but relevant understanding, and zero if wrong.\n"
        "- Justify each score concisely and remain consistent across all answers.\n"
        "- Avoid hallucinations or invented details.\n"
        "- Finally, list conceptual weaknesses (topics not understood) and recurring mistakes "
        "(e.g., 'calculation errors', 'definition confusion').\n\n"
        "Return ONLY a single JSON object in this structure:\n"
        "{\n"
        "  \"student_info\": {\n"
        "     \"name\": str,\n"
        "     \"registration_number\": str,\n"
        "     \"course_code\": str\n"
        "  },\n"
        "  \"answers\": [\n"
        "     {\"question_number\": int, \"marks_awarded\": float, \"max_marks\": float, \"justification\": str}\n"
        "  ],\n"
        "  \"conceptual_weaknesses\": [str],\n"
        "  \"common_mistakes\": [str]\n"
        "}\n"
        "Output must be valid JSON with no markdown, no explanations outside the structure."
    )

# 🧩 --- UPDATED CORE CALL ---
def ask_gemini_for_structured_extraction(
    client: genai.Client,
    student_file: Any,
    qp_or_key_file: Optional[Any],
    model: str = PRIMARY_MODEL,
    max_output_tokens: int = 8192,
) -> str:
    """
    Pass both student file and question paper/answer key to the model.
    """
    user_prompt = build_grading_prompt()
    contents = [user_prompt, student_file]
    if qp_or_key_file:
        contents.append(qp_or_key_file)

    def _call():
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=0.0,
            ),
        )
        return resp

    resp = retry_backoff(_call)
    return resp.text

def parse_json_from_model_text(text: str) -> Dict:
    """
    Cleans Gemini's response and extracts valid JSON even if wrapped in code fences or text.
    """
    cleaned = text.strip()

    # Remove markdown-style fences (```json ... ```)
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # If model output accidentally includes explanation or cutoff text,
    # try to extract the JSON part safely.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        cleaned = cleaned[start:end+1]
    else:
        raise ValueError("No valid JSON structure found in model output.")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Save raw output for inspection if parsing fails
        with open("debug_raw_output.txt", "w", encoding="utf-8") as dbg:
            dbg.write(text)
        logger.info("\n⚠️ JSON parsing failed. Raw output saved to debug_raw_output.txt for inspection.")
        raise e


# ---------------------------
# CLI runner
# ---------------------------
def main(student_pdf_path: str, qp_or_key_path: Optional[str] = None, api_key: Optional[str] = None):
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError("Google API key not provided. Pass it as argument or set GOOGLE_API_KEY env var.")

    client = create_client(api_key)

    student_file = upload_file(client, student_pdf_path)
    qp_or_key_file = None
    if qp_or_key_path and os.path.exists(qp_or_key_path):
        qp_or_key_file = upload_file(client, qp_or_key_path)

    logger.info("Requesting intelligent grading from model:", PRIMARY_MODEL)
    try:
        raw = ask_gemini_for_structured_extraction(client, student_file, qp_or_key_file, model=PRIMARY_MODEL)
    except Exception as e:
        logger.info("Primary model failed:", e)
        for fb in FALLBACK_MODELS:
            logger.info("Trying fallback model:", fb)
            try:
                raw = ask_gemini_for_structured_extraction(client, student_file, qp_or_key_file, model=fb)
                logger.info("Fallback succeeded with", fb)
                break
            except Exception as e2:
                logger.info("Fallback", fb, "failed:", e2)
        else:
            raise RuntimeError("All models failed; aborting") from e

    logger.info("Raw model response (truncated):", raw[:1000])
    parsed = parse_json_from_model_text(raw)
    out_file = os.path.join(OUT_DIR, os.path.basename(student_pdf_path) + "_graded.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(parsed, fh, indent=2, ensure_ascii=False)
    logger.info("Parsed JSON written to:", out_file)
    return parsed

# if __name__ == "__main__":
#     student_pdf_path = "D:\EvalAI\data\papers\paper data 1.pdf"
#     qp_or_key_path = "D:\EvalAI\data\papers\qp.pdf"  # can contain key or just questions
#     print(f"Running on: {student_pdf_path}")
#     main(student_pdf_path, qp_or_key_path, API_KEY)

