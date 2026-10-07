# Packs the app into a container so it runs the same way on any computer or server.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Train inside the container so the model matches the installed scikit-learn.
RUN python -m app.train_model

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
