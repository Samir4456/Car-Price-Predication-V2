FROM python:3.12.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY ./code /app

COPY ./code/Model_New/model.pkl /app/Model_New/model.pkl

EXPOSE 5000

CMD ["python", "app.py"]
