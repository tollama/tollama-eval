"""Generate examples/sample.csv for end-to-end testing."""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2020-01-01", periods=60, freq="D")

rows = []
for uid in ["s1", "s2"]:
    base = 100.0 if uid == "s1" else 200.0
    for i, d in enumerate(dates):
        seasonal = 5.0 * np.sin(2 * np.pi * i / 7)
        noise = rng.normal(0, 1.0)
        rows.append(
            {
                "unique_id": uid,
                "ds": d.strftime("%Y-%m-%d"),
                "y": round(base + i * 0.5 + seasonal + noise, 2),
            }
        )

df = pd.DataFrame(rows)
df.to_csv("examples/sample.csv", index=False)
print(f"Created examples/sample.csv with {len(df)} rows")
