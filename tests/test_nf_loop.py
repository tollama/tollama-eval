import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
df = pd.DataFrame({"unique_id": ["1"]*20, "ds": pd.date_range("2020-01-01", periods=20), "y": list(range(20))})

print("Fold 1...")
model = NHITS(h=2, input_size=4, max_steps=10, accelerator="cpu")
nf1 = NeuralForecast(models=[model], freq="D")
nf1.fit(df)

print("Fold 2...")
model2 = NHITS(h=2, input_size=4, max_steps=10, accelerator="cpu")
nf2 = NeuralForecast(models=[model2], freq="D")
nf2.fit(df)

print("Done folds!")
