"""Evidence offsets must survive HTML removal and Unicode decoding."""

import pytest

from scripts.prepare_nq import eligible_annotations, extract_text, mapped_span, page_key, records


def test_source_offsets_survive_html_entities_and_block_boundaries():
    tokens = [
        {"token": "<P>", "is_html": True},
        {"token": "Caf&#233;", "is_html": False},
        {"token": "<b>", "is_html": True},
        {"token": "&amp;", "is_html": False},
        {"token": "</b>", "is_html": True},
        {"token": "tea", "is_html": False},
        {"token": "</P>", "is_html": True},
        {"token": "<P>", "is_html": True},
        {"token": "Next", "is_html": False},
    ]
    text, mapping = extract_text(tokens)
    assert text == "Café & tea\n\nNext"
    evidence = mapped_span({"start_token": 0, "end_token": 7}, mapping, text)
    assert evidence["text"] == "Café & tea"
    answer = mapped_span({"start_token": 3, "end_token": 6}, mapping, text)
    assert text[answer["start_char"]:answer["end_char"]] == "& tea"
    assert mapping[0] is None


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (0, 3), (0, 1)])
def test_invalid_or_html_only_source_spans_are_rejected(start, end):
    with pytest.raises(ValueError):
        mapped_span({"start_token": start, "end_token": end}, [None, [0, 1]], "x")


def test_page_grouping_ignores_revision_protocol_and_fragment():
    assert page_key("http://en.wikipedia.org/wiki/Caf%C3%A9?oldid=1#History") == page_key(
        "https://en.wikipedia.org/wiki/Café?oldid=2"
    )
    assert page_key("https://en.wikipedia.org//w/index.php?title=Caf%C3%A9&amp;oldid=3") == page_key(
        "https://en.wikipedia.org/wiki/Café"
    )


def test_columnar_sequences_reject_truncated_fields():
    with pytest.raises(ValueError, match="lengths"):
        records({"start_token": [1, 2], "end_token": [3]})


def test_selection_requires_two_paragraph_votes_and_keeps_multispan_answers():
    annotation = {
        "id": "a",
        "long_answer": {"start_token": 0, "end_token": 4},
        "short_answers": {"start_token": [1, 2], "end_token": [2, 3], "text": ["a", "b"]},
        "yes_no_answer": -1,
    }
    row = {"document": {"tokens": {"token": ["<P>", "a", "b", "</P>"]}},
           "annotations": [annotation]}
    assert eligible_annotations(row) == []
    row["annotations"] = [annotation, dict(annotation, id="b")]
    assert len(eligible_annotations(row)) == 2
    assert len(records(eligible_annotations(row)[0]["short_answers"])) == 2
    row["document"]["tokens"]["token"][0] = "<Table>"
    assert eligible_annotations(row) == []


def test_boolean_conflicts_are_not_silently_scored_as_extractive_answers():
    row = {"document": {"tokens": {"token": []}},
           "annotations": [{"yes_no_answer": 0}]}
    assert eligible_annotations(row) == []
