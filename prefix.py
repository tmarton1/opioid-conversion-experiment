import pandas as pd

MODEL_PREFIXES = {
    "gpt-4.1-mini": "openai/",
    "gpt-3.5-turbo": "openai/",
    "claude-3-5-haiku-20241022": "anthropic/",
    "claude-3.5-haiku": "anthropic/",
    "claude-3.5-sonnet:20241022": "anthropic/",
    "llama-3.3-70b-versatile": "meta-llama/",
    "llama-3.3-70b-instruct": "meta-llama/",
    "mistral-large-2407": "mistralai/",
    "mistralai/mistral-large-2407": "",
    "mixtral-8x22b-instruct": "mistralai/",
    "mistralai/mixtral-8x22b-instruct": "",
    "gemini-2.5-pro": "google/",
    "google/gemini-2.5-pro": "",
    "gemini-2.5-flash": "google/",
    "google/gemini-2.5-flash": "",
    "command-r-plus-08-2024": "cohere/",
    "cohere/command-r-plus-08-2024": "",
    "jamba-mini-1.7": "ai21/",
    "ai21/jamba-mini-1.7": "",
    "jamba-large-1.7": "ai21/",
    "ai21/jamba-large-1.7": "",
    "qwen-2.5-72b-instruct": "qwen/",
    "qwen/qwen-2.5-72b-instruct": "",
    "deepseek-chat": "deepseek/",
    "deepseek/deepseek-chat": "",
    "models/gemini-2.5-flash": "google/",
}

def harmonize_model_names(csv_path, output_path=None, model_column="model"):
    """
    Ensures all model names in the CSV have the correct prefix.
    Writes to output_path if provided, otherwise overwrites the input file.
    """
    df = pd.read_csv(csv_path)
    def harmonize(name):
        # Remove any existing prefix
        base = name.split('/')[-1]
        # Find canonical prefix
        for key, prefix in MODEL_PREFIXES.items():
            if base == key.split('/')[-1]:
                return prefix + base if prefix else name
        return name  # fallback: leave unchanged if not found

    df[model_column] = df[model_column].apply(harmonize)
    out = output_path if output_path else csv_path
    df.to_csv(out, index=False)
    print(f"Harmonized model names written to {out}")