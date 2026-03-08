import pandas as pd
import json
from pathlib import Path

def main():
    repo_dir = Path("/Users/yongchoelchoi/Documents/GitHub/tollama/hf_data")
    # let's find an index
    with open(repo_dir / "_index.json") as f:
        index = json.load(f)
    
    target_id = "An-j96/SuperstoreData"
    target_item = None
    for item in index["items"]:
        if item.get("hf_id") == target_id:
            target_item = item
            break
            
    if target_item is None:
        # Fallback to the first retail dataset
        for item in index["items"]:
            if item.get("industry_hint") == "retail":
                target_item = item
                break
                
    if target_item is None:
        target_item = index["items"][0]
        
    raw_file = repo_dir / target_item["raw_file"]
    meta_file = repo_dir / target_item["meta_file"]
    
    with open(meta_file) as f:
        meta = json.load(f)
        
    ts_col = meta.get("timestamp_column", "timestamp")
    tgt_col = meta.get("target_column", "target")
    
    rows = []
    with open(raw_file) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
                
    df = pd.DataFrame(rows)
    print("Cols:", df.columns)
    
    # find robust ts col and tgt col
    if ts_col not in df.columns:
        ts_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), df.columns[0])
    if tgt_col not in df.columns:
        tgt_col = next((c for c in df.columns if c.lower() in ("sales", "y", "target", "value")), df.columns[-1])
        
    df = df.rename(columns={ts_col: "ds", tgt_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"], format=None, errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    
    # drop nas
    df = df.dropna(subset=["ds", "y"])
    
    # group by Date
    agg_df = df.groupby("ds")["y"].sum().reset_index()
    agg_df = agg_df.sort_values("ds")
    
    # format ds nicely back
    agg_df["ds"] = agg_df["ds"].dt.strftime("%Y-%m-%d")
    agg_df["unique_id"] = "superstore_sales"
    
    # reorder
    agg_df = agg_df[["unique_id", "ds", "y"]]
    
    agg_df.to_csv("real_data.csv", index=False)
    print("Saved real_data.csv with", len(agg_df), "daily aggregated rows")
if __name__ == "__main__":
    main()
