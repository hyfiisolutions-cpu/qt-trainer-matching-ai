import pandas as pd

SKILL_ALIASES = {
    "react.js": "react", "reactjs": "react", "nodejs": "node.js", "node js": "node.js",
    "springboot": "spring boot", "postgres": "postgresql", "postgres sql": "postgresql",
    "ml": "machine learning", "dl": "deep learning", "gen ai": "generative ai",
    "genai": "generative ai", "powerbi": "power bi",
}

LOCATION_ALIASES = {
    "bengaluru": "bangalore", "new delhi": "delhi", "gurugram": "gurgaon", "mysuru": "mysore"
}

def load_data(trainers_path="trainer_matching/dataset/trainers.csv", requirements_path="trainer_matching/dataset/requirements.csv"):
    return pd.read_csv(trainers_path), pd.read_csv(requirements_path)

def normalize_skill(skill):
    if pd.isna(skill):
        return ""
    skill = " ".join(str(skill).strip().lower().split())
    return SKILL_ALIASES.get(skill, skill)

def parse_skills(skill_string):
    if pd.isna(skill_string):
        return []
    result = [normalize_skill(s) for s in str(skill_string).split(",")]
    return list(dict.fromkeys([s for s in result if s]))

def normalize_location(location):
    if pd.isna(location):
        return ""
    loc = str(location).strip().lower()
    return LOCATION_ALIASES.get(loc, loc)

def normalize_availability(value):
    if pd.isna(value):
        return 0
    return 1 if str(value).strip().lower() in {"yes", "available", "true", "1"} else 0

def clean_trainers(df):
    df = df.copy()
    df["skills_list"] = df["skills"].apply(parse_skills)
    df["location_clean"] = df["location"].apply(normalize_location)
    df["available_binary"] = df["available"].apply(normalize_availability)
    df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce").fillna(0).clip(lower=0)
    df["hourly_rate"] = pd.to_numeric(df["hourly_rate"], errors="coerce").fillna(0).clip(lower=0)
    return df

def clean_requirements(df):
    df = df.copy()
    df["required_skills_list"] = df["required_skills"].apply(parse_skills)
    df["location_clean"] = df["location"].apply(normalize_location)
    for col in ["min_experience", "duration_months", "budget_per_hour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    return df

def preprocess_data(trainers, requirements):
    return clean_trainers(trainers), clean_requirements(requirements)

if __name__ == "__main__":
    trainers, requirements = load_data()
    trainers, requirements = preprocess_data(trainers, requirements)
    print(f"Preprocessed {len(trainers)} trainers and {len(requirements)} requirements.")
