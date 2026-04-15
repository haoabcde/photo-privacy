FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV HOST=0.0.0.0
ENV PORT=5001
ENV PHOTO_PRIVACY_DATA_DIR=/data

EXPOSE 5001

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5001} wsgi:app --workers 1 --threads 4 --timeout 120"]

