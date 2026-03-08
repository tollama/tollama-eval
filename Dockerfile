FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY src/ src/

RUN uv pip install --system --no-cache ".[dev]"

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/ts-autopilot /usr/local/bin/ts-autopilot
COPY --from=builder /app/src /app/src

RUN useradd --create-home autopilot
USER autopilot

ENTRYPOINT ["ts-autopilot"]
CMD ["--help"]
