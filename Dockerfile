FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY app ./app
RUN pip install --upgrade pip && pip install -e .
COPY data ./data
EXPOSE 8000 8501
CMD ["uvicorn","heart_ai.api:app","--host","0.0.0.0","--port","8000"]
