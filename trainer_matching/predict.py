import argparse
from preprocessing import load_data, preprocess_data
from matching_model import rank_trainers

def main():
    parser = argparse.ArgumentParser(description="QT Trainer Matching AI")
    parser.add_argument("--requirement", default="R001")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--method", choices=["overlap", "tfidf"], default="overlap")
    args = parser.parse_args()

    trainers, requirements = preprocess_data(*load_data())
    match = requirements[requirements["requirement_id"] == args.requirement]
    if match.empty:
        print(f"Requirement {args.requirement} not found.")
        return
    requirement = match.iloc[0]
    print("=" * 65)
    print("QT TRAINER MATCHING AI")
    print("=" * 65)
    print(f"Requirement: {requirement['requirement_id']} | Skills: {requirement['required_skills']} | Experience: {requirement['min_experience']}+ years | Location: {requirement['location']} | Budget: Rs.{requirement['budget_per_hour']}/hr")

    results = rank_trainers(trainers, requirement, top_k=args.top_k, method=args.method)
    print(f"\nMethod: {args.method.upper()}\n")
    for i, r in enumerate(results, 1):
        print(f"{i:>2}. {r['trainer_name']:<12} {r['final_percentage']:>6.2f}%  | Skill {r['skill_score']*100:5.1f}% | Exp {r['experience_score']*100:5.1f}% | Loc {r['location_score']*100:5.1f}% | Budget {r['budget_score']*100:5.1f}%")

if __name__ == "__main__":
    main()
