from src.retrieval.retriever import tokenize


def test_trailing_punctuation_is_dropped():
    assert tokenize("WBC, platelets: low.") == ["wbc", "platelets", "low"]


def test_lowercases():
    assert tokenize("HbA1c") == tokenize("hba1c") == ["hba1c"]


def test_patient_id_kept_whole_inside_parentheses():
    tokens = tokenize("Patient ID (SYN-2024-001)")
    assert "syn-2024-001" in tokens
    assert "(syn-2024-001)" not in tokens


def test_query_id_matches_document_id():
    # the whole point: an ID typed in a query must produce the same token
    assert "syn-2024-001" in tokenize("SYN-2024-001")


def test_decimals_kept_and_not_split():
    tokens = tokenize("HbA1c: 9.2%. Creatinine 1.8")
    assert "9.2" in tokens and "1.8" in tokens
    assert "9" not in tokens and "2" not in tokens


def test_sentence_final_number_loses_its_period():
    assert tokenize("Follow up in 3.") == ["follow", "up", "in", "3"]


def test_dose_string_stays_intact():
    assert "5mg" in tokenize("Apixaban 5mg twice daily")


def test_hyphen_and_slash_compounds_also_emit_parts():
    assert tokenize("post-op") == ["post-op", "post", "op"]
    assert tokenize("nausea/vomiting") == ["nausea/vomiting", "nausea", "vomiting"]
    assert "mg/dl" in tokenize("180 mg/dL")


def test_run_together_list_items():
    # MTSamples transcriptions often lose spaces between numbered items
    assert tokenize("Diabetes.,3. Hypertension") == ["diabetes", "3", "hypertension"]


def test_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize(" -- | ** ") == []
