# Single image: PostgreSQL 16 + FastAPI (OpenEnv) for Hugging Face Spaces / demos.
# linux/amd64: matches HF Space GPU; requirements.txt uses CUDA 12.4 torch wheels
# (Apple Silicon: `docker build --platform=linux/amd64` so pip can resolve +cu124).
FROM --platform=linux/amd64 postgres:16-bookworm

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-full python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Upgrade pip so legacy resolver quirks and long backtracks on large dep trees are less likely.
RUN pip3 install -U --no-cache-dir --break-system-packages pip setuptools wheel \
    && pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .
COPY docker/entrypoint-sqlsage.sh /usr/local/bin/entrypoint-sqlsage.sh
RUN chmod +x /usr/local/bin/entrypoint-sqlsage.sh \
    && chown -R postgres:postgres /app

ENV POSTGRES_PASSWORD=sqlsage
ENV POSTGRES_DB=sqlsage
ENV POSTGRES_USER=postgres
ENV POSTGRES_HOST=127.0.0.1
ENV POSTGRES_PORT=5432

EXPOSE 7860

ENTRYPOINT ["/usr/local/bin/entrypoint-sqlsage.sh"]
