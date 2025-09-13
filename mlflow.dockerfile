FROM python:3.12.11-slim

WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow[extras]

EXPOSE 5001

# Run MLflow server (all args in one JSON array)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001", "--backend-store-uri", "/mlruns", "--default-artifact-root", "/mlruns"]
