FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/vqa/.cache/huggingface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8502

WORKDIR /app

RUN useradd --create-home --uid 10001 vqa \
    && mkdir -p /home/vqa/.cache/huggingface /app/artifacts \
    && chown -R vqa:vqa /home/vqa /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0" \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=vqa:vqa app.py ./app.py
COPY --chown=vqa:vqa src ./src
COPY --chown=vqa:vqa artifacts/flickr_vqa_patch_attention_10000.pt ./artifacts/flickr_vqa_patch_attention_10000.pt

USER vqa
EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8502"]