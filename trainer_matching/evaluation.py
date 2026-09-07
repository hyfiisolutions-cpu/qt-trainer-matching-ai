import pandas as pd
from preprocessing import load_data, preprocess_data
from matching_model import rank_trainers


def create_ground_truth(trainers, requirements):
    """Synthetic business-rule relevance labels for prototype evaluation only."""
    gt = {}
    for _, req in requirements.iterrows():
        req_skills = set(req["required_skills_list"])
        relevant = []
        for _, tr in trainers.iterrows():
            if tr["available_binary"] == 0 or not req_skills:
                continue
            overlap = len(req_skills & set(tr["skills_list"])) / len(req_skills)
            within_reasonable_budget = tr["hourly_rate"] <= req["budget_per_hour"] * 1.20
            if overlap >= 0.50 and tr["experience_years"] >= req["min_experience"] and within_reasonable_budget:
                relevant.append(tr["trainer_id"])
        gt[req["requirement_id"]] = relevant
    return gt


def hit_at_k(ranked, relevant, k):
    return int(any(x in relevant for x in ranked[:k]))


def reciprocal_rank(ranked, relevant):
    for i, x in enumerate(ranked, 1):
        if x in relevant:
            return 1 / i
    return 0.0


def evaluate_method(trainers, requirements, ground_truth, method):
    rows, h1, h3, rrs = [], [], [], []
    for _, req in requirements.iterrows():
        relevant = ground_truth.get(req["requirement_id"], [])
        if not relevant:
            continue
        ranked = [x["trainer_id"] for x in rank_trainers(trainers, req, method=method)]
        a, b, c = hit_at_k(ranked, relevant, 1), hit_at_k(ranked, relevant, 3), reciprocal_rank(ranked, relevant)
        h1.append(a); h3.append(b); rrs.append(c)
        rows.append({"requirement_id": req["requirement_id"], "method": method, "hit_at_1": a, "hit_at_3": b, "reciprocal_rank": round(c, 4), "top_trainer": ranked[0] if ranked else None})
    metrics = {
        "method": method,
        "requirements_evaluated": len(h1),
        "hit_at_1": sum(h1)/len(h1) if h1 else 0,
        "hit_at_3": sum(h3)/len(h3) if h3 else 0,
        "mrr": sum(rrs)/len(rrs) if rrs else 0,
    }
    return metrics, rows


if __name__ == "__main__":
    trainers, requirements = preprocess_data(*load_data())
    gt = create_ground_truth(trainers, requirements)
    all_rows, summaries = [], []
    for method in ["overlap", "tfidf"]:
        metrics, rows = evaluate_method(trainers, requirements, gt, method)
        summaries.append(metrics); all_rows.extend(rows)

    summary_df = pd.DataFrame(summaries)
    details_df = pd.DataFrame(all_rows)
    summary_df.to_csv("evaluation_summary.csv", index=False)
    details_df.to_csv("evaluation_details.csv", index=False)

    print("\nMODEL COMPARISON")
    print("=" * 60)
    for m in summaries:
        print(f"{m['method'].upper():<10} | Cases: {m['requirements_evaluated']:>2} | Hit@1: {m['hit_at_1']*100:6.2f}% | Hit@3: {m['hit_at_3']*100:6.2f}% | MRR: {m['mrr']:.4f}")
    print("\nSaved evaluation_summary.csv and evaluation_details.csv")
