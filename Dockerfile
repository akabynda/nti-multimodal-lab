FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs
RUN pip install --no-deps .
ENV NTI_PROJECT_ROOT=/app NTI_DATA_ROOT=/data PYTHONUNBUFFERED=1
EXPOSE 8765
CMD ["uvicorn", "nti_lab.app:app", "--host", "0.0.0.0", "--port", "8765"]
