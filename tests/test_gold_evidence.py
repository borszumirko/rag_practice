from scripts.check_gold_evidence import messages, score, summarize


def test_scoring_handles_variants_and_repeated_tokens():
    assert score("The Republic of Tecala.", ["Republic of Tecala"])["exact_match"] == 1
    assert score("red red blue", ["red blue blue"])["token_f1"] == 2 / 3
    assert score("Paris", ["London", "Paris"])["exact_match"] == 1
    assert score("", ["Paris"]) == {"exact_match": 0, "token_f1": 0}


def test_prompt_contains_only_question_and_deduplicated_evidence():
    reference = {"answer_text": "SECRET ANSWER", "evidence": {
        "start_char": 0, "end_char": 8, "text": "Evidence"}}
    prompt = messages("A question?", {"references": [reference, reference]})
    assert "SECRET ANSWER" not in str(prompt)
    assert prompt[1]["content"].count("Evidence\n") == 1
    assert "A question?" in prompt[1]["content"]


def test_missing_cost_is_reported_instead_of_claiming_free_usage():
    report = summarize([{"usage": {}, "exact_match": 0, "token_f1": 0,
                         "latency_ms": 1, "finish_reason": "stop"}], 1)
    assert report["responses_missing_cost"] == 1
