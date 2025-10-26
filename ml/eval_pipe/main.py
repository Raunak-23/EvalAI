from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from dotenv import load_dotenv
from ml.eval_pipe.gemini_vlm import main as grade_main

# Load environment variables
load_dotenv()

app = FastAPI(
    title="EvalAI Grading API",
    description="Uploads student and question PDFs, evaluates using Gemini, and returns structured JSON.",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "Welcome to EvalAI Grading API. Use /docs to test the API."}

@app.post("/grade")
async def grade_pdfs(
    student_pdf: UploadFile = File(..., description="Student answer sheet PDF"),
    qp_or_key_pdf: UploadFile = File(..., description="Question paper or key PDF")
):
    """
    Accepts two PDFs → Runs Gemini-based grading → Returns JSON result.
    """
    try:
        # Save uploaded PDFs to temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as s_tmp:
            s_tmp.write(await student_pdf.read())
            student_path = s_tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as q_tmp:
            q_tmp.write(await qp_or_key_pdf.read())
            qp_path = q_tmp.name

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set in environment")

        # Call your grading function
        result = grade_main(student_path, qp_path, api_key)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary files
        for p in [student_path, qp_path]:
            if os.path.exists(p):
                os.remove(p)
