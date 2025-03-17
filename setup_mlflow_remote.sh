#!/bin/bash
# Script to start MLflow server with configuration from config/config.yaml

# Configuration
CONFIG_FILE="config/config.yaml"
DB_PATH="./mlflow.db"
ARTIFACT_STORE="./mlruns"

# Default values in case parsing fails
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=5000

echo "Reading MLflow configuration from $CONFIG_FILE..."

# Extract host and port from config/config.yaml using Python
# This avoids complex YAML parsing in bash
MLFLOW_CONFIG=$(python3 -c "
import yaml
import sys
from urllib.parse import urlparse

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    if 'mlflow' in config and 'tracking_uri' in config['mlflow']:
        uri = config['mlflow']['tracking_uri']
        parsed = urlparse(uri)
        host = parsed.hostname or '$DEFAULT_HOST'
        port = parsed.port or $DEFAULT_PORT
        print(f'{host} {port}')
    else:
        print('$DEFAULT_HOST $DEFAULT_PORT')
except Exception:
    print('$DEFAULT_HOST $DEFAULT_PORT')
")

# Split the output into host and port
HOST=$(echo $MLFLOW_CONFIG | cut -d' ' -f1)
PORT=$(echo $MLFLOW_CONFIG | cut -d' ' -f2)

# Make sure directories exist
mkdir -p $(dirname "$DB_PATH")
mkdir -p "$ARTIFACT_STORE"

echo "Starting MLflow server on $HOST:$PORT"
echo "Database: sqlite:///$DB_PATH"
echo "Artifacts: $ARTIFACT_STORE"

# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///$DB_PATH \
  --default-artifact-root $ARTIFACT_STORE \
  --host $HOST \
  --port $PORT