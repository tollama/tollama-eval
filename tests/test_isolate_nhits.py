from ts_autopilot.pipeline import run_from_csv
try:
    print("Running pipeline manually from script...")
    run_from_csv("examples/sample.csv", output_dir="out", model_names=["NHITS"], n_folds=3, horizon=14)
    print("DONE pipeline!")
except Exception as e:
    import traceback
    traceback.print_exc()
