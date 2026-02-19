import pandas as pd

class ModelBatchMeansBuilder:
    def __init__(self, pair_model_csv="pair_model_analysis.csv"):
        self.df = pd.read_csv(pair_model_csv)

    def filter_df(self, ome, include_nielsen=None, include_examples=None, explain_units=None, has_vignette_text=None):
        df = self.df.copy()
        if ome is not None:
            df = df[df["ome"] == ome]
        if include_nielsen is not None:
            df = df[df["include_nielsen"] == include_nielsen]
        if include_examples is not None and "include_examples" in df.columns:
            df = df[df["include_examples"] == include_examples]
        if explain_units is not None and "explain_units" in df.columns:
            df = df[df["explain_units"] == explain_units]
        if has_vignette_text is not None and "has_vignette_text" in df.columns:
            df = df[df["has_vignette_text"] == has_vignette_text]
        return df

    def get_range_col(self, prefix):
        for col in self.df.columns:
            if col.startswith(prefix):
                return col
        raise ValueError(f"No column found for prefix '{prefix}'")

    def build_means_df(self, combinations):
        tight_col = self.get_range_col("tight(")
        broad_col = self.get_range_col("broad(")
        rows = []
        for combo in combinations:
            batch_name = combo["batch_name"]
            ome = combo.get("ome", None)
            include_nielsen = combo.get("include_nielsen", None)
            include_examples = combo.get("include_examples", None)
            explain_units = combo.get("explain_units", None)
            has_vignette_text = combo.get("has_vignette_text", None)
            filtered = self.filter_df(ome, include_nielsen, include_examples, explain_units, has_vignette_text)
            for model in filtered["model"].unique():
                model_df = filtered[filtered["model"] == model]
                mean_weight = model_df["weight"].mean()
                mean_in_tight = model_df[tight_col].mean()
                mean_in_broad = model_df[broad_col].mean()
                rows.append({
                    "model": model,
                    "batch_name": batch_name,
                    "ome": ome,
                    "include_nielsen": include_nielsen,
                    "include_examples": include_examples,
                    "explain_units": explain_units,
                    "has_vignette_text": has_vignette_text,
                    "mean_weight": mean_weight,
                    "mean_in_tight": mean_in_tight,
                    "mean_in_broad": mean_in_broad
                })
        return pd.DataFrame(rows)

    def save_means_df(self, combinations, output_csv):
        means_df = self.build_means_df(combinations)
        means_df.to_csv(output_csv, index=False)
        print(f"✓ Saved batch means to {output_csv}")