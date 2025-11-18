# syntax=docker/dockerfile:1

############################
# Stage 1: build React app #
############################
FROM node:20-alpine AS web-build
WORKDIR /web
COPY web/package*.json ./
RUN npm install
COPY web ./
RUN npm run build

############################
# Stage 2: Python API      #
############################
FROM python:3.11-slim AS webapi
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential curl && \
	rm -rf /var/lib/apt/lists/*

# Python deps
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r ./api/requirements.txt

# App code
COPY api ./api

# Copy built frontend into /app/static (served by FastAPI)
RUN mkdir -p /app/static
COPY --from=web-build /web/dist /app/static

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
