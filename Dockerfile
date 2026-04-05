# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

WORKDIR /workspace

RUN pip install poetry

COPY pyproject.toml /workspace/spectrum-news-agent/
COPY --from=agent_monitoring pyproject.toml /workspace/agent-monitoring/
COPY --from=agent_monitoring README.md /workspace/agent-monitoring/
COPY --from=agent_monitoring agent_monitoring /workspace/agent-monitoring/agent_monitoring

WORKDIR /workspace/spectrum-news-agent

RUN poetry config virtualenvs.create false \
    && poetry install --no-root

COPY . /workspace/spectrum-news-agent

WORKDIR /workspace/spectrum-news-agent

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
