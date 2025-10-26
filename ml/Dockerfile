# -------------------------------
# Base Image
# -------------------------------
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the ML code and requirements first
COPY ./ml /app/ml
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create directory for results
RUN mkdir -p /app/ml/eval_pipe/graded_results

# Expose FastAPI port (Hugging Face uses 7860, local can use 8000)
EXPOSE 7860

# Environment variable (key injected from compose or cloud)
ENV GOOGLE_API_KEY=""

# Run the API
CMD ["uvicorn", "ml.eval_pipe.main:app", "--host", "0.0.0.0", "--port", "7860"]
