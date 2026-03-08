import pandas as pd

def fit_nhits():
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    df = pd.DataFrame({"unique_id": ["1"]*20, "ds": pd.date_range("2020-01-01", periods=20), "y": list(range(20))})
    model = NHITS(h=2, input_size=4, max_steps=10, accelerator="cpu")
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)
    print("Fit completed")

fit_nhits()
