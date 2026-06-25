from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from typing import Any

from aiterate.domain import DatasetSnapshot


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_raw_data(name: str, raw_data: str) -> DatasetSnapshot:
    text = raw_data.strip()
    cases = _parse_json_cases(text) or _parse_yaml_cases(text) or _parse_csv_cases(text) or _split_text_cases(text)
    return DatasetSnapshot(
        name=name,
        raw_text=text,
        normalized_cases=cases,
        content_hash=stable_hash(text),
    )


def _parse_json_cases(text: str) -> list[str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return _cases_from_payload(payload)


def _parse_yaml_cases(text: str) -> list[str]:
    if not _looks_like_yaml(text):
        return []
    try:
        import yaml
    except ImportError:
        return []
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    return _cases_from_payload(payload)


def _cases_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [_serialize_case(item) for item in payload]
    if isinstance(payload, dict):
        rows = payload.get("cases") or payload.get("examples") or payload.get("data") or payload.get("records")
        if isinstance(rows, list):
            return [_serialize_case(item) for item in rows]
        return [_serialize_case(payload)]
    return []


def _serialize_case(item: Any) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, sort_keys=True)


def _parse_csv_cases(text: str) -> list[str]:
    if "," not in text or "\n" not in text:
        return []
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    except csv.Error:
        return []
    if not reader.fieldnames or len(reader.fieldnames) < 2:
        return []
    if any(None in row for row in rows):
        return []
    if any(value is None for row in rows for value in row.values()):
        return []
    return [json.dumps(row, sort_keys=True) for row in rows if any(row.values())]


def _split_text_cases(text: str) -> list[str]:
    chunks = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if part.strip()]
    return chunks or ([text] if text else [])


def _looks_like_yaml(text: str) -> bool:
    return bool(re.search(r"(^|\n)\s*[-\w]+\s*:", text) or re.search(r"(^|\n)\s*-\s+\w+", text))
