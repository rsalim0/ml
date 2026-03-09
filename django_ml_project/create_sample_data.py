#!/usr/bin/env python3
"""Generate synthetic vehicles_ml_dataset.csv for training and EDA."""
import os
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "dummy-data", "vehicles_ml_dataset.csv")

# Rwanda's 30 districts
DISTRICTS = [
    "Gasabo", "Kicukiro", "Nyarugenge", "Bugesera", "Gatsibo", "Kayonza",
    "Kirehe", "Ngoma", "Nyagatare", "Rwamagana", "Burera", "Gakenke",
    "Gicumbi", "Musanze", "Rulindo", "Gisagara", "Huye", "Kamonyi",
    "Muhanga", "Nyamagabe", "Nyanza", "Nyaruguru", "Ruhango", "Karongi",
    "Ngororero", "Nyabihu", "Nyamasheke", "Rubavu", "Rutsiro", "Rusizi",
]

INCOME_LEVELS = ["Low", "Medium", "High"]
random.seed(42)

n = 1000
data = {
    "client_name": [f"Client_{i+1}" for i in range(n)],
    "district": [random.choice(DISTRICTS) for _ in range(n)],
    "year": [random.randint(2015, 2024) for _ in range(n)],
    "kilometers_driven": [random.randint(5000, 150000) for _ in range(n)],
    "seating_capacity": [random.choice([2, 4, 5, 7, 9, 14]) for _ in range(n)],
    "estimated_income": [round(random.uniform(200000, 5000000), 2) for _ in range(n)],
}
# selling_price correlated with year, km, income
data["selling_price"] = [
    round(
        (data["year"][i] - 2010) * 500000
        - data["kilometers_driven"][i] * 2
        + data["estimated_income"][i] * 0.15
        + random.gauss(0, 200000),
        2,
    )
    for i in range(n)
]
# Clamp selling_price to positive
for i in range(n):
    data["selling_price"][i] = max(500000, min(8000000, data["selling_price"][i]))

# income_level from estimated_income terciles
sorted_income = sorted(data["estimated_income"])
t1 = sorted_income[n // 3]
t2 = sorted_income[2 * n // 3]
data["income_level"] = [
    "Low" if data["estimated_income"][i] < t1
    else "High" if data["estimated_income"][i] >= t2
    else "Medium"
    for i in range(n)
]

df = pd.DataFrame(data)
df.to_csv(OUT_PATH, index=False)
print(f"Created {OUT_PATH} with {len(df)} rows")
