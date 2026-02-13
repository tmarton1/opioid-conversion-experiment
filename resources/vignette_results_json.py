import pandas as pd
import json

def save_vignette_results_json(csv_file="scenario_experiment_results.csv", json_file="vignette_model_results.json"):
    df = pd.read_csv(csv_file)
    df["include_nielsen"] = df["include_nielsen"].astype(bool)

    def extract_parts(prompt_with, prompt_without):
        if prompt_with.startswith(prompt_without):
            extra = prompt_with[len(prompt_without):].strip()
            if extra:
                return f"{prompt_without} ({extra})"
            else:
                return prompt_with
        else:
            return prompt_with

    result_list = []
    for vignette_id in df["vignette_id"].unique():
        group = df[df["vignette_id"] == vignette_id]
        prompt_with = group[group["include_nielsen"] == True]["prompt"].iloc[0]
        prompt_without = group[group["include_nielsen"] == False]["prompt"].iloc[0]
        scenario = extract_parts(prompt_with, prompt_without)
        expected_to_dose = group["expected_to_dose"].iloc[0] if "expected_to_dose" in group else None
        observed_to_dose = []
        for model, model_group in group.groupby("model"):
            without_nielsen = model_group[model_group["include_nielsen"] == False]["observed_to_dose"].tolist()
            with_nielsen = model_group[model_group["include_nielsen"] == True]["observed_to_dose"].tolist()
            observed_to_dose.append({
                "model": model,
                "without_nielsen": without_nielsen,
                "with_nielsen": with_nielsen
            })
        result_list.append({
            "scenario": scenario,
            "expected_to_dose": expected_to_dose,
            "observed_to_dose": observed_to_dose
        })

    with open(json_file, "w") as f:
        json.dump(result_list, f, indent=2)