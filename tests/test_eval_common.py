"""Key-fact matcher and refusal detection used by the eval scripts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_common import contains_fact, is_refusal, key_fact_recall, strip_citations  # noqa: E402

from src.agent.grounding import REFUSAL  # noqa: E402


@pytest.mark.parametrize("answer", [
    "Lymph nodes were normal in size.",
    "No known drug allergies.",
    "Renal function and liver enzymes are normal; data from the 2023 visit agree.",
    "Metastases were not found on the CT scan described in the imaging report.",
    "Nodes: none enlarged. Known history of AF.",
])
def test_ordinary_clinical_text_is_not_a_refusal(answer):
    assert not is_refusal(answer)


@pytest.mark.parametrize("answer", [
    REFUSAL,
    "The answer to this question is not found in the available clinical documents.",
    "This information is not mentioned anywhere in the provided passages.",
    "The patient's BMI isn’t documented in the records.",
    "There is no information about the patient's BMI.",
    "The documents do not state the patient's blood type.",
    "The report doesn’t contain any data on smoking history.",
])
def test_refusals_are_detected(answer):
    assert is_refusal(answer)


def test_citation_tags_do_not_count_as_facts():
    answer = "Apixaban 5 mg BID [Source: report_006_stroke.pdf | MEDICATIONS]"
    assert key_fact_recall(answer, ["stroke"]) == 0
    assert key_fact_recall(answer, ["Apixaban", "5 mg"]) == 1
    for tagged in ("x 【Source: report_006_stroke.pdf | Section: MEDICATIONS】",
                   "x [report_006_stroke.pdf | MEDICATIONS]",
                   "x (see report_006_stroke.pdf)"):
        assert "stroke" not in strip_citations(tagged)


def test_numbers_inside_dates_and_ids_do_not_match():
    assert not contains_fact("DOB: 1942-02-28", "28")
    assert not contains_fact("Seen 28/02/2024", "02")
    assert not contains_fact("Patient ID: SYN-2024-028", "28")
    assert not contains_fact("Hb 14.8 g/dL", "4.8")
    assert not contains_fact("Bisoprolol 2.5 mg", "5 mg")


@pytest.mark.parametrize("text, fact", [
    ("Bisoprolol 5 mg OD", "5 mg"),
    ("HbA1c 9.2%.", "9.2%"),
    ("WBC 11.2 x 10^9/L", "11.2"),
    ("EF 70-75%", "70-75"),
    ("EF 70-75%", "70"),
    ("EF 70-75%", "75"),
    ("BP 120/80", "120"),
    ("Patient ID: SYN-2024-001", "SYN-2024-001"),
    ("Age: 28, female", "28"),
    ("(28)", "28"),
    ("ages 28 - 30", "28"),
    ("37weeks gestation", "37 weeks"),
    ("AF with RVR", "AF"),
])
def test_legitimate_numbers_still_match(text, fact):
    assert contains_fact(text, fact)


def test_markdown_and_thousands_separators_are_ignored():
    assert contains_fact("a CURB-65 score of **1**, mild", "score of 1")
    assert contains_fact("estimated fetal weight 3,289 grams", "3289")
    # A comma between separate numbers in a list is not a thousands separator.
    assert not contains_fact("doses of 5,10 and 20 mg", "510")
