# --- builder ---
FROM python:3.11-slim AS builder
WORKDIR /app
COPY inference_requirements.txt .
RUN pip install --no-cache-dir -r inference_requirements.txt
COPY infer.py model.pth ./
COPY utils/ utils/

# --- runtime ---
FROM python:3.11-slim
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app
WORKDIR /app 
ENV PORT=8080

RUN useradd --create-home apiuser
USER apiuser

CMD ["uvicorn","infer:app","--host","0.0.0.0","--port","8080"]
    