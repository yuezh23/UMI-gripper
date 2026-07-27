import pandas as pd

TOP_K = 100

for modality in ["force", "gelsight", "combined"]:

    importance = pd.read_csv(
        f"artifacts/force_state/within_object_full_v3/within_object/{modality}/feature_importance.csv"
    )

    top = (
        importance
        .sort_values("importance", ascending=False)
        .head(TOP_K)["feature"]
    )

    top.to_csv(
        f"artifacts/force_state/within_object_full_v3/top100_{modality}_features.csv",
        index=False,
        header=False
    )

    print(f"{modality}: saved {len(top)} features.")
    print(top.head(10).to_list())