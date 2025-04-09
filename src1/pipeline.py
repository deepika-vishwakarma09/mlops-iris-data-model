import os

# Define all the steps to run in order
steps = [
    "src1/data_loader.py",
    "src1/data_injection.py",
    "src1/data_preprocessing.py",
    "src1/featureengineering.py",
    "src1/model_training.py",
    "src1/model_evaluation.py"
]

print("🚀 Starting the MLOps Iris Pipeline...\n")

for step in steps:
    print(f"\n▶️ Running: {step}")
    exit_code = os.system(f"python {step}")
    if exit_code != 0:
        print(f"❌ Step failed: {step}")
        break
    else:
        print(f"✅ Completed: {step}")

print("\n✅ All steps completed (if no errors above).")
