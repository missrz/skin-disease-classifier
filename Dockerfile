# Use the official slim Python image
FROM python:3.11-slim

# 1) Install OS deps (if you need apt packages, add them here)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2) Set workdir and copy just requirements
WORKDIR /home/appuser/app
COPY requirements.txt .

# 3) Install all Python deps as root, so uvicorn ends up in /usr/local/bin
RUN pip install --no-cache-dir -r requirements.txt

# 4) Create appuser and chown the app folder
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /home/appuser

# 5) Switch to non-root
USER appuser

# 6) Copy your model and code (ownership already set)
COPY --chown=appuser:appuser model.pth class_idx.json main.py ./

# 7) Expose and run
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
