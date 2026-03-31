from glitchlab.controller_utils import calculate_quality_score


class DummyState:
    def __init__(self, debug_attempts: int):
        self.debug_attempts = debug_attempts


def test_calculate_quality_score_returns_documented_fields_and_values():
    result = calculate_quality_score({"total_tokens": 60000}, DummyState(debug_attempts=2))

    assert set(result.keys()) == {"score", "tokens_used", "debug_attempts"}
    assert result["tokens_used"] == 60000
    assert result["debug_attempts"] == 2
    assert result["score"] == 78
