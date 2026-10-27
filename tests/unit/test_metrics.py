from leakguard.datagen.hf import gold_for, merge_adjacent, split_of
from leakguard.eval.failure_analysis import masked
from leakguard.eval.metrics import leaked_values, span_prf, summarise


def rec(**kw):
    base = {"gold_action": "allow", "gold_protect": False, "final": "allow", "output_changed": False, "gold_spans_n": 0,
            "leaked_n": 0, "tp": 0, "fp": 0, "fn": 0, "latency_ms": 1.0, "scenario": "s", "lang": "en", "citations": ["DEF-001"]}
    base.update(kw)
    return base


def test_span_prf_category_aware():
    gold = [{"start": 0, "end": 5, "entity": "EMAIL"}, {"start": 10, "end": 20, "entity": "AWS_ACCESS_KEY"}]
    pred = [{"start": 1, "end": 4, "entity": "PHONE"}, {"start": 12, "end": 18, "entity": "GENERIC_SECRET"},
            {"start": 30, "end": 35, "entity": "PERSON"}]
    # EMAIL~PHONE match (both pii), AWS~GENERIC_SECRET match (both secret), PERSON is a false positive
    assert span_prf(gold, pred) == (2, 1, 0)


def test_leakage_is_end_to_end():
    text = "mail jane@acme.in now"
    gold = [{"start": 5, "end": 17, "entity": "EMAIL"}]
    assert leaked_values(text, gold, None) == []  # blocked -> nothing leaked
    assert leaked_values(text, gold, "mail <EMAIL_1> now") == []
    assert leaked_values(text, gold, text) == gold


def test_summary_rates():
    recs = [rec(), rec(final="block"), rec(final="redact"),
            rec(gold_action="redact", gold_protect=True, final="redact", gold_spans_n=2, leaked_n=1, tp=1, fn=1)]
    s = summarise(recs, valid_citations={"DEF-001"})
    assert s["false_block_rate"] == round(1 / 3, 4)
    assert s["over_redaction_rate"] == round(1 / 3, 4)
    assert s["leakage_recall"] == 0.5
    assert s["groundedness"] == 1.0


def test_gold_spec():
    assert gold_for([{"entity": "PRIVATE_KEY"}], "prompt", "external") == "block"
    assert gold_for([{"entity": "EMAIL"}], "prompt", "internal") == "allow"
    assert gold_for([{"entity": "GOV_ID"}], "prompt", "internal") == "redact"
    assert gold_for([{"entity": "EMAIL"}] * 8, "prompt", "external") == "escalate"
    assert gold_for([{"entity": "GITHUB_TOKEN"}], "response", "external") == "block"
    assert gold_for([{"entity": "PHONE"}], "tool_payload", "external") == "escalate"
    assert gold_for([], "tool_payload", "external") == "allow"


def test_split_is_deterministic():
    assert split_of("abc") == split_of("abc")
    assert {split_of(str(i)) for i in range(200)} == {"train", "dev", "test"}


def test_merge_adjacent_names():
    text = "Hi Asha Rao!"
    spans = merge_adjacent([{"start": 3, "end": 7, "entity": "PERSON"}, {"start": 8, "end": 11, "entity": "PERSON"}], text)
    assert spans == [{"start": 3, "end": 11, "entity": "PERSON"}]


def test_failure_report_masks_values():
    assert masked("key AKIAXXXX end", [{"start": 4, "end": 12, "entity": "AWS_ACCESS_KEY"}]) == "key [AWS_ACCESS_KEY] end"
