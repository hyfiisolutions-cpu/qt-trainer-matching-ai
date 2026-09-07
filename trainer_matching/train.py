from preprocessing import load_data, preprocess_data
from matching_model import WEIGHTS

if __name__ == "__main__":
    trainers, requirements = preprocess_data(*load_data())
    print("QT Trainer Matching Baseline")
    print(f"Trainers: {len(trainers)}")
    print(f"Requirements: {len(requirements)}")
    print("Weights:", WEIGHTS)
    print("No supervised model is trained in this baseline; ranking is feature-based and reproducible.")
