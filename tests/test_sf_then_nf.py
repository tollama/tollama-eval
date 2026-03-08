import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

df = pd.DataFrame({"unique_id": ["1"]*20, "ds": pd.date_range("2020-01-01", periods=20), "y": list(range(20))})

# 1. StatsForecast
sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=1)
sf.fit(df)
print("StatsForecast Done!")

# 2. NeuralForecast
model = NHITS(h=2, input_size=4, max_steps=10, accelerator="cpu")
nf = NeuralForecast(models=[model], freq="D")
nf.fit(df)
print("NeuralForecast Done!")
