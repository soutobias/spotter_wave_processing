FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml .
COPY requirements.txt .
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/
COPY entrypoint.sh /app/

RUN pip install --upgrade pip \
    && pip install -r requirements.txt
RUN pip install .

VOLUME /data

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
