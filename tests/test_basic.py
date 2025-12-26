"""Basic tests for ticket-triage-ml package."""


def test_package_import():
    """Test that the package can be imported."""
    import ticket_triage_ml

    assert ticket_triage_ml.__version__ == "0.1.0"


def test_data_schema_import():
    """Test that data schemas can be imported."""
    from ticket_triage_ml.data.schema import InferenceInput

    input_obj = InferenceInput(text="Test ticket text")
    assert input_obj.text == "Test ticket text"


def test_inference_output_schema():
    """Test InferenceOutput schema validation."""
    from ticket_triage_ml.data.schema import InferenceOutput

    output = InferenceOutput(
        topic="billing",
        priority="high",
        topic_scores={"billing": 0.8, "support": 0.2},
        priority_scores={"low": 0.1, "medium": 0.2, "high": 0.7},
    )

    assert output.topic == "billing"
    assert output.priority == "high"
    assert output.topic_scores["billing"] == 0.8


def test_paths_module():
    """Test paths module functions."""
    from ticket_triage_ml.utils.paths import get_project_root

    project_root = get_project_root()
    assert project_root.exists()
    assert (project_root / "pyproject.toml").exists()
