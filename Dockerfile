# Single image: PostgreSQL 16 + FastAPI (OpenEnv) for Hugging Face Spaces / demos.
FROM postgres:16-bookworm

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-full python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

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
