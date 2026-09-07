# QT Trainer Matching AI

## Objective
Build a reproducible trainer recommendation baseline that ranks available trainers against a training requirement using skills, experience, location, availability, and hourly-rate compatibility.

## Dataset
- 100 synthetic QT-style trainers
- 30 synthetic requirements
- Fields include skills, experience, city, availability, rate, duration, and budget

## Pipeline
Requirement → Preprocessing → Skill Similarity → Experience → Location → Availability → Budget → Weighted Match Score → Ranked Trainers

## Scoring
- Skill similarity: 40%
- Experience: 20%
- Location: 15%
- Availability: 15%
- Budget: 10%

Two skill baselines are included:
1. Exact required-skill overlap
2. TF-IDF cosine similarity

## Installation
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python train.py
python predict.py --requirement R001 --top-k 5 --method overlap
python predict.py --requirement R001 --top-k 5 --method tfidf
pytest -q
python evaluation.py
```

## Evaluation
`evaluation.py` reports Hit@1, Hit@3, and Mean Reciprocal Rank (MRR) for both baselines and writes:
- `evaluation_summary.csv`
- `evaluation_details.csv`

The current relevance labels are **synthetic business-rule labels**, not historical QT recruiter selections. This is suitable for prototype evaluation only.

## Limitations
- Synthetic data rather than historical QT data
- Manually selected feature weights
- Simplified city proximity logic
- Availability is binary
- TF-IDF does not understand deep semantic equivalence between skills
- No learning-to-rank model yet

## Next Improvements
- Replace synthetic labels with anonymized historical recruiter selections
- Add sentence-transformer embeddings and skill ontology/aliases
- Learn ranking weights from successful trainer assignments
- Add real geographic distance and trainer calendars
- Add recruiter feedback loop and model monitoring
