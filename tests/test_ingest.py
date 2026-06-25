from aiterate.ingest import normalize_raw_data


def test_normalize_raw_text_creates_cases_and_hash():
    dataset = normalize_raw_data("demo", "First case. Second case.")
    assert dataset.content_hash
    assert len(dataset.normalized_cases) == 2


def test_normalize_json_cases():
    dataset = normalize_raw_data("demo", '{"cases": [{"input": "a"}, {"input": "b"}]}')
    assert len(dataset.normalized_cases) == 2


def test_normalize_yaml_cases():
    dataset = normalize_raw_data(
        "demo",
        """
cases:
  - input: Summarize this refund policy.
    expected: Cite the source.
  - input: Data is missing.
    expected: Escalate uncertainty.
""",
    )
    assert len(dataset.normalized_cases) == 2
    assert "refund policy" in dataset.normalized_cases[0]


def test_normalize_csv_cases():
    dataset = normalize_raw_data("demo", "input,expected\nhello,cite source\nbye,escalate\n")
    assert len(dataset.normalized_cases) == 2
