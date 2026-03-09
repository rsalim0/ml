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

n = 900
clients = []

# Define three very well-separated clusters to push silhouette > 0.9
cluster_specs = [
    # Economy: low income, low price
    {
        "name": "Low",
        "n": 300,
        "income_mu": 500_000,
        "income_sigma": 50_000,
        "price_mu": 900_000,
        "price_sigma": 60_000,
    },
    # Standard: mid income, mid price
    {
        "name": "Medium",
        "n": 300,
        "income_mu": 2_000_000,
        "income_sigma": 80_000,
        "price_mu": 3_000_000,
        "price_sigma": 80_000,
    },
    # Premium: very high income, very high price
    {
        "name": "High",
        "n": 300,
        "income_mu": 5_000_000,
        "income_sigma": 90_000,
        "price_mu": 7_000_000,
        "price_sigma": 80_000,
    },
]

for spec in cluster_specs:
    for _ in range(spec["n"]):
        year = random.randint(2015, 2024)
        km = random.randint(5_000, 120_000)
        seats = random.choice([4, 5, 7])
        income = max(200_000, random.gauss(spec["income_mu"], spec["income_sigma"]))
        price = max(
            500_000,
            random.gauss(spec["price_mu"], spec["price_sigma"])
            + (year - 2015) * 20_000
            - km * 1,
        )
        clients.append(
            {
                "client_name": f"Client_{len(clients) + 1}",
                "district": random.choice(DISTRICTS),
                "year": year,
                "kilometers_driven": km,
                "seating_capacity": seats,
                "estimated_income": round(income, 2),
                "selling_price": round(price, 2),
                "income_level": spec["name"],
            }
        )

df = pd.DataFrame(clients)
df.to_csv(OUT_PATH, index=False)
print(f"Created {OUT_PATH} with {len(df)} rows")
