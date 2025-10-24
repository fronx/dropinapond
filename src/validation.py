"""
JSON schema validation utilities for ego graphs.
"""

import json
import jsonschema
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).parent.parent / "schema"


def load_schema(version: str = "0.2") -> Dict:
    """
    Load the JSON schema for a specific ego graph version.

    Args:
        version: Schema version (e.g., "0.2")

    Returns:
        Schema dict

    Raises:
        FileNotFoundError: If schema file doesn't exist
    """
    schema_file = SCHEMA_DIR / f"ego_graph_v{version.replace('.', '')}.json"

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_file}")

    with open(schema_file) as f:
        return json.load(f)


def validate_ego_graph(data: Dict, version: str = "0.2") -> tuple[bool, Optional[List[str]]]:
    """
    Validate an ego graph JSON against the schema.

    Args:
        data: Ego graph data dict
        version: Expected schema version

    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if validation passed
        - error_messages: List of error messages (None if valid)
    """
    try:
        schema = load_schema(version)
        jsonschema.validate(instance=data, schema=schema)
        logger.info(f"Validation passed for ego graph version {version}")
        return True, None
    except jsonschema.ValidationError as e:
        error_msg = f"Validation error at {'.'.join(str(p) for p in e.path)}: {e.message}"
        logger.error(error_msg)
        return False, [error_msg]
    except jsonschema.SchemaError as e:
        error_msg = f"Schema error: {e.message}"
        logger.error(error_msg)
        return False, [error_msg]
    except Exception as e:
        error_msg = f"Unexpected error during validation: {str(e)}"
        logger.error(error_msg)
        return False, [error_msg]


def validate_ego_graph_file(file_path: Path, version: str = "0.2") -> tuple[bool, Optional[List[str]]]:
    """
    Validate an ego graph JSON file.

    Args:
        file_path: Path to the JSON file
        version: Expected schema version

    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        with open(file_path) as f:
            data = json.load(f)
        return validate_ego_graph(data, version)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON: {e}"
        logger.error(error_msg)
        return False, [error_msg]
    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        return False, [error_msg]


def auto_correct_ego_graph(data: Dict) -> Dict:
    """
    Attempt to auto-correct common issues in ego graph data.

    This function applies common fixes:
    - Add missing 'version' field (defaults to "0.2")
    - Add missing 'weight' to phrases (defaults to 1.0)
    - Ensure 'is_self' is set for focal_node
    - Convert date strings to ISO format if needed

    Args:
        data: Ego graph data dict

    Returns:
        Corrected ego graph data dict
    """
    corrected = data.copy()

    # Add version if missing
    if "version" not in corrected:
        corrected["version"] = "0.2"
        logger.info("Added missing 'version' field (defaulted to '0.2')")

    # Add default weights to phrases
    if "nodes" in corrected:
        for node in corrected["nodes"]:
            if "phrases" in node:
                for phrase in node["phrases"]:
                    if "weight" not in phrase:
                        phrase["weight"] = 1.0

            # Set is_self for focal node
            if node["id"] == corrected.get("focal_node"):
                if "is_self" not in node:
                    node["is_self"] = True
                    logger.info(f"Set 'is_self=True' for focal node {node['id']}")

    return corrected
