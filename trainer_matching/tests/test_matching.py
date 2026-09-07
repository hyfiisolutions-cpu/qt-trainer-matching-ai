from matching_model import (
    calculate_skill_similarity, calculate_tfidf_skill_similarity,
    calculate_experience_score, calculate_location_score,
    calculate_availability_score, calculate_budget_score, calculate_final_score
)

def test_perfect_skill_match():
    assert calculate_skill_similarity(["java", "spring boot"], ["java", "spring boot"]) == 1.0

def test_partial_skill_match():
    assert round(calculate_skill_similarity(["java", "spring boot", "hibernate"], ["java", "spring boot"]), 2) == 0.67

def test_no_skill_match():
    assert calculate_skill_similarity(["java"], ["python"]) == 0.0

def test_tfidf_identical_is_high():
    assert calculate_tfidf_skill_similarity(["java", "spring boot"], ["java", "spring boot"]) > 0.99

def test_experience():
    assert calculate_experience_score(4, 6) == 1.0
    assert calculate_experience_score(4, 2) == 0.5

def test_locations():
    assert calculate_location_score("bangalore", "bangalore") == 1.0
    assert calculate_location_score("bangalore", "mysore") == 0.7

def test_availability():
    assert calculate_availability_score("Yes") == 1.0
    assert calculate_availability_score("No") == 0.0

def test_budget():
    assert calculate_budget_score(2000, 1500) == 1.0
    assert calculate_budget_score(2000, 2500) == 0.8

def test_final_score_perfect():
    assert calculate_final_score(1, 1, 1, 1, 1) == 1.0
