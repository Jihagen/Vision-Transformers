from repuap.tasks import JudgmentTask, parse_response, extract_last_json_block


def _numeric_task():
    return JudgmentTask(valid_labels=[1, 2, 3, 4, 5], scale_width=4.0,
                         prompt_builder=lambda ctx: "irrelevant", json_key="label", label_type=int)


def test_extract_last_json_block_picks_last_valid():
    text = 'garbage {"a": 1} more text {"label": 3, "explanation": "ok"}'
    assert extract_last_json_block(text) == {"label": 3, "explanation": "ok"}


def test_parse_valid_json():
    task = _numeric_task()
    r = parse_response('{"label": 3, "explanation": "It is moderately relevant."}', task)
    assert r.label == 3
    assert r.explanation == "It is moderately relevant."
    assert r.parse_ok is True
    assert r.used_fallback is False


def test_parse_truncated_json_falls_back_to_regex():
    task = _numeric_task()
    r = parse_response('Sure, here is the JSON:\n{"label": 4, "explan', task)
    assert r.label == 4
    assert r.parse_ok is True
    assert r.used_fallback is True
    assert r.explanation is None


def test_parse_out_of_range_label_rejected():
    task = _numeric_task()
    r = parse_response('{"label": 7, "explanation": "nonsense"}', task)
    assert r.label is None
    assert r.parse_ok is False


def test_parse_garbage_text_rejected():
    task = _numeric_task()
    r = parse_response("I cannot answer that.", task)
    assert r.label is None
    assert r.parse_ok is False
    assert r.used_fallback is False


def test_parse_zero_label_not_treated_as_falsy():
    task = JudgmentTask(valid_labels=[0, 1, 2], scale_width=2.0,
                         prompt_builder=lambda ctx: "x", json_key="label")
    r = parse_response('{"label": 0, "explanation": "no damage visible"}', task)
    assert r.label == 0
    assert r.parse_ok is True


def test_string_label_task():
    task = JudgmentTask(valid_labels=["yes", "no"], scale_width=1.0,
                         prompt_builder=lambda ctx: "x", json_key="relevant", label_type=str,
                         label_to_score={"yes": 1.0, "no": 0.0})
    r = parse_response('{"relevant": "yes", "explanation": "matches"}', task)
    assert r.label == "yes"
    assert r.parse_ok is True
    assert task.score(r.label) == 1.0


def test_anchor_default_quotes_string_labels_only():
    numeric_task = JudgmentTask(valid_labels=[1, 2], scale_width=1.0,
                                 prompt_builder=lambda ctx: "x", json_key="label", label_type=int)
    string_task = JudgmentTask(valid_labels=["yes", "no"], scale_width=1.0,
                                prompt_builder=lambda ctx: "x", json_key="relevant", label_type=str)
    assert numeric_task.anchor() == '{"label":'
    assert string_task.anchor() == '{"relevant":"'


def test_score_identity_for_numeric_labels():
    task = _numeric_task()
    assert task.score(4) == 4.0


def test_score_raises_without_map_for_string_labels():
    task = JudgmentTask(valid_labels=["a", "b"], scale_width=1.0,
                         prompt_builder=lambda ctx: "x", json_key="label", label_type=str)
    import pytest
    with pytest.raises(ValueError):
        task.score("a")
