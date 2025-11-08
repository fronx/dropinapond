#!/usr/bin/env python3
"""Resume Neo4j Aura instance using credentials from .env"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

neo4j_id = os.getenv('NEO4J_ID')
client_id = os.getenv('NEO4J_CLIENT_ID')
client_secret = os.getenv('NEO4J_CLIENT_SECRET')

if not all([neo4j_id, client_id, client_secret]):
    print("Error: NEO4J_ID, NEO4J_CLIENT_ID, and NEO4J_CLIENT_SECRET must be set in .env")
    sys.exit(1)

print(f"Resuming Neo4j instance {neo4j_id}...")

# Try to add credentials, ignore if they already exist
result = subprocess.run([
    "uv", "run", "aura", "credentials", "add",
    "--name", "auto",
    "--client-id", client_id,
    "--client-secret", client_secret
], capture_output=True, text=True)

# If credentials exist, that's fine - just use them
if result.returncode != 0 and "already exist" not in result.stderr:
    print(f"Error adding credentials: {result.stderr}")
    sys.exit(1)

# Set the credentials to use
subprocess.run([
    "uv", "run", "aura", "credentials", "use", "auto"
], capture_output=True, text=True)

# Resume the instance
print(f"Resuming instance {neo4j_id}...")
result = subprocess.run([
    "uv", "run", "aura", "instances", "resume",
    "--instance-id", neo4j_id,
    "--wait"
], capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error resuming instance: {result.stderr}")
    sys.exit(1)

print(result.stdout)
print(f"\n✓ Instance {neo4j_id} is resuming (takes 1-2 minutes)")
print("Run the backend server once it's ready:")
print("  uv run uvicorn server.main:app --reload --port 3002")
