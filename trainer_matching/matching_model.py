from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import parse_skills

WEIGHTS = {"skill": 0.40, "experience": 0.20, "location": 0.15, "availability": 0.15, "budget": 0.10}
NEARBY_CITIES = {
    "bangalore": {"mysore"}, "mysore": {"bangalore"},
    "delhi": {"noida", "gurgaon"}, "noida": {"delhi", "gurgaon"}, "gurgaon": {"delhi", "noida"},
    "mumbai": {"pune"}, "pune": {"mumbai"},
}

def calculate_skill_similarity(required_skills, trainer_skills):
    required = set(parse_skills(required_skills) if isinstance(required_skills, str) else required_skills)
    trainer = set(parse_skills(trainer_skills) if isinstance(trainer_skills, str) else trainer_skills)
    return 0.0 if not required else len(required & trainer) / len(required)

def calculate_tfidf_skill_similarity(required_skills, trainer_skills):
    req = " ".join(required_skills) if isinstance(required_skills, list) else str(required_skills)
    tr = " ".join(trainer_skills) if isinstance(trainer_skills, list) else str(trainer_skills)
    if not req.strip() or not tr.strip():
        return 0.0
    matrix = TfidfVectorizer().fit_transform([req, tr])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])

def calculate_experience_score(required_experience, trainer_experience):
    required_experience, trainer_experience = float(required_experience), float(trainer_experience)
    if required_experience <= 0:
        return 1.0
    return 1.0 if trainer_experience >= required_experience else trainer_experience / required_experience

def calculate_location_score(required_location, trainer_location):
    req, tr = str(required_location).strip().lower(), str(trainer_location).strip().lower()
    if req == tr:
        return 1.0
    return 0.7 if tr in NEARBY_CITIES.get(req, set()) else 0.3

def calculate_availability_score(available):
    if isinstance(available, str):
        return 1.0 if available.strip().lower() in {"yes", "available", "true", "1"} else 0.0
    return 1.0 if available else 0.0

def calculate_budget_score(budget, trainer_rate):
    budget, trainer_rate = float(budget), float(trainer_rate)
    if budget <= 0 or trainer_rate <= 0:
        return 0.0
    return 1.0 if trainer_rate <= budget else budget / trainer_rate

def calculate_final_score(skill_score, experience_score, location_score, availability_score, budget_score):
    return (
        WEIGHTS["skill"] * skill_score + WEIGHTS["experience"] * experience_score +
        WEIGHTS["location"] * location_score + WEIGHTS["availability"] * availability_score +
        WEIGHTS["budget"] * budget_score
    )

def _match(trainer, requirement, method="overlap"):
    skill_score = calculate_skill_similarity(requirement["required_skills_list"], trainer["skills_list"]) if method == "overlap" else calculate_tfidf_skill_similarity(requirement["required_skills_list"], trainer["skills_list"])
    experience_score = calculate_experience_score(requirement["min_experience"], trainer["experience_years"])
    location_score = calculate_location_score(requirement["location_clean"], trainer["location_clean"])
    availability_score = calculate_availability_score(trainer["available_binary"])
    budget_score = calculate_budget_score(requirement["budget_per_hour"], trainer["hourly_rate"])
    final_score = calculate_final_score(skill_score, experience_score, location_score, availability_score, budget_score)
    return {
        "trainer_id": trainer["trainer_id"], "trainer_name": trainer["name"],
        "skill_score": skill_score, "experience_score": experience_score,
        "location_score": location_score, "availability_score": availability_score,
        "budget_score": budget_score, "final_score": final_score,
        "final_percentage": round(final_score * 100, 2), "method": method,
    }

def rank_trainers(trainers, requirement, top_k=None, method="overlap"):
    results = [_match(t, requirement, method) for _, t in trainers.iterrows() if t["available_binary"] == 1]
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k] if top_k else results

def rank_trainers_tfidf(trainers, requirement, top_k=None):
    return rank_trainers(trainers, requirement, top_k=top_k, method="tfidf")
