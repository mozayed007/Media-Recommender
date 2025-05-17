# Docker and Milvus Setup

This document describes the Docker setup used for running the Milvus vector database and its dependencies for the Media Recommender project.

## Overview

The project uses Docker Compose (`docker-compose.yml`) to manage the following services:

1. **Milvus Standalone (`standalone`)**: The core vector database service (v2.4.1). It handles data storage, indexing, and searching.
    * Exposes port `19530` for SDK connections.
    * Exposes port `9091` for metrics.
    * Depends on `etcd` and `minio`.
    * Stores persistent data in `./volumes/milvus`.
2. **etcd (`etcd`)**: A distributed key-value store used by Milvus for metadata management.
    * Stores persistent data in `./volumes/etcd`.
3. **MinIO (`minio`)**: An S3-compatible object storage service used by Milvus for storing large data chunks (like index files).
    * Exposes port `9000` for the MinIO API.
    * Exposes port `9001` for the MinIO Console (Web UI).
    * Stores persistent data in `./volumes/minio`.
4. **Attu (`attu`)**: A web-based GUI tool for managing and inspecting the Milvus database.
    * Exposes port `8000` on the host machine, mapping to port `3000` inside the container.
    * Connects to the `standalone` service using the internal Docker network address `milvus-standalone:19530`.
    * Depends on `standalone`.

All services are connected via a custom Docker network named `milvus`.

## Usage

### Starting Services

To start all services in detached mode (running in the background), navigate to the project's root directory (where `docker-compose.yml` is located) in your terminal and run:

```bash
docker-compose up -d
```

### Stopping Services

To stop all running services:

```bash
docker-compose down
```

### Accessing Services

* **Milvus SDK**: Connect using the host `localhost` and port `19530`.
* **MinIO Console**: Open `http://localhost:9001` in your web browser. Use `minioadmin` / `minioadmin` for credentials.
* **Attu GUI**: Open `http://localhost:8000` in your web browser. It should automatically connect to the Milvus instance.

### Data Persistence

Service data (Milvus database files, etcd metadata, MinIO objects) is persisted in the `./volumes` directory on the host machine. This ensures data is not lost when containers are stopped and restarted.
