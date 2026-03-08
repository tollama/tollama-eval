import pandas as pd
import sys
import os

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

print("Creating dataframe...")
df = pd.DataFrame({"unique_id": ["1"]*20, "ds": pd.date_range("2020-01-01", periods=20), "y": list(range(20))})

print("Instantiating NHITS...")
model = NHITS(h=2, input_size=4, max_steps=10, accelerator="cpu")

print("Instantiating NeuralForecast...")
nf = NeuralForecast(models=[model], freq="D")

print("Fitting...")
nf.fit(df)
print("Done!")
