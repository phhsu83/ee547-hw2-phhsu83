FROM python:3.11-slim

# Install PyTorch (CPU only for smaller image)
#RUN pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.0.1

WORKDIR /app
COPY train_embeddings.py /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "/app/train_embeddings.py"]