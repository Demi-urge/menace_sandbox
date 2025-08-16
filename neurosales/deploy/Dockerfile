FROM python:3.11-slim
ARG HEAVY_DEPS=true
WORKDIR /app
COPY requirements.txt requirements-extra.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$HEAVY_DEPS" = "true" ]; then \
        pip install --no-cache-dir -r requirements-extra.txt && \
        playwright install --with-deps; \
    fi

COPY . .

RUN if [ "$HEAVY_DEPS" = "true" ]; then \
        python scripts/setup_heavy_deps.py --download-only; \
    fi

CMD ["uvicorn", "neurosales.api_gateway:create_app", "--host", "0.0.0.0", "--port", "8000"]
