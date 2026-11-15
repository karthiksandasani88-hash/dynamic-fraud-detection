"""Generate reproducible synthetic transaction data for the demo."""
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    n = 6000
    amount = rng.lognormal(3.5, 1.0, n).clip(1, 5000)
    hour = rng.integers(0, 24, n)
    distance = rng.gamma(2.0, 12.0, n).clip(0, 300)
    count = rng.poisson(2.2, n).clip(0, 25)
    merchant_risk = rng.beta(2.0, 5.0, n)
    score = (-5.0 + 0.0007 * amount + 0.075 * distance
             + 0.23 * count + 3.0 * merchant_risk
             + 1.0 * ((hour <= 4) | (hour >= 23)).astype(float))
    probability = 1 / (1 + np.exp(-score))
    labels = rng.binomial(1, probability)
    df = pd.DataFrame({
        "amount": amount.round(2),
        "hour": hour,
        "distance_from_home_km": distance.round(2),
        "transactions_last_24h": count,
        "merchant_risk_score": merchant_risk.round(4),
        "is_fraud": labels,
    })
    output = Path(__file__).parent / "data" / "transactions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} synthetic transactions to {output}")
    print(f"Synthetic fraud rate: {df.is_fraud.mean():.1%}")

if __name__ == "__main__":
    main()
