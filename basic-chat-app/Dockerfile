FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 APP_HOST=0.0.0.0 APP_PORT=8000 RAG_DATA_DIR=/app/data
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY static ./static
RUN useradd --create-home appuser && mkdir -p /app/data /app/.models && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --start-period=300s --interval=30s --timeout=5s CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/api/config', timeout=4)"
CMD ["python", "app.py"]
