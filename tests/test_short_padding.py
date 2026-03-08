import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

df = pd.DataFrame({"unique_id": ["1"]*18, "ds": pd.date_range("2020-01-01", periods=18), "y": list(range(18))})

print("Testing NHITS with small dataframe (18 rows) and large window (28+14) AND start_padding_enabled=True...")
model = NHITS(h=14, input_size=28, max_steps=10, accelerator="cpu", start_padding_enabled=True)

nf = NeuralForecast(models=[model], freq="D")
nf.fit(df)
print("Fit succeeded?!")
