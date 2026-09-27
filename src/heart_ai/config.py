from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
DATA_PATH=PROJECT_ROOT/"data"/"heart.csv"
TARGET="HeartDisease"
NUMERICAL=["Age","RestingBP","Cholesterol","FastingBS","MaxHR","Oldpeak"]
CATEGORICAL=["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"]
ALL_FEATURES=NUMERICAL+CATEGORICAL
RANDOM_STATE=42
TEST_SIZE=0.2
