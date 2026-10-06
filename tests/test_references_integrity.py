"""Guard against overwritten foundational citation entries."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _bib_entry(key: str) -> str:
    text = (ROOT / "references.bib").read_text(encoding="utf-8")
    marker = "{" + key + ","
    start = text.index(marker)
    next_entry = text.find("\n@", start + 1)
    return text[start: next_entry if next_entry != -1 else len(text)]


def test_foundational_reference_keys_match_cited_methods():
    expected_fragments = {
        "damodaran2018deepjdot": ["DeepJDOT", "Unsupervised Domain Adaptation"],
        "long2018cada": ["Conditional Adversarial Domain Adaptation", "Long"],
        "10918592": ["Learning to Compare", "Relation Network"],
        "kirkpatrick2017overcoming": ["Overcoming catastrophic forgetting", "Proceedings of the National Academy"],
        "de2021continual": ["Continual Learning Survey", "De Lange"],
        "wang2022continual": ["Continual Test-Time Domain Adaptation", "Wang"],
        "niu2022efficient": ["Efficient Test-Time Model Adaptation without Forgetting", "Niu"],
        "yuan2023robust": ["Robust Test-Time Adaptation in Dynamic Scenarios", "Yuan"],
    }

    for key, fragments in expected_fragments.items():
        entry = _bib_entry(key)
        for fragment in fragments:
            assert fragment in entry


def test_ctta_light_methods_cite_original_papers_in_manuscript():
    text = (ROOT / "main_revision.tex").read_text(encoding="utf-8")

    assert "CoTTA-light\\cite{wang2022continual}" in text
    assert "EATA-light\\cite{niu2022efficient}" in text
    assert "RoTTA-light\\cite{yuan2023robust}" in text
    assert "CoTTA-light\\cite{wang2022continual} uses" in text
    assert "EATA-light\\cite{niu2022efficient} keeps" in text
    assert "RoTTA-light\\cite{yuan2023robust} maintains" in text


def test_footprint_table_avoids_ambiguous_tta_update_wording():
    text = (ROOT / "main_revision.tex").read_text(encoding="utf-8")

    assert "head state above" not in text
    assert "TTA update & Same as MLP head" in text
    assert "8,576 MACs/sample (forward)" in text
    assert "TTA operates on the existing MLP head without adding extra model parameters" in text
    assert "does not include backward/optimizer operations" in text
