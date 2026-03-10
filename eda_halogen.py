import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import os


def run_eda():
    os.makedirs("eda_results", exist_ok=True)

    print("Downloading HALoGEN dataset...")
    # Load dataset
    dataset = load_dataset("lasha-nlp/HALoGEN-prompts", split="train")
    df = pd.DataFrame(dataset)

    print("\n--- Basic Dataset Info ---")
    print(f"Total Prompts: {len(df)}")


    print(f"Actual Dataset Columns: {df.columns.tolist()}")

    category_col = next((col for col in ['Category', 'category', 'task', 'Task', 'domain'] if col in df.columns),
                        df.columns[1])
    prompt_col = next((col for col in ['prompt', 'Prompt', 'text', 'question'] if col in df.columns), df.columns[0])

    print(f"--> Using '{category_col}' for categories and '{prompt_col}' for prompts.\n")

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y=category_col, order=df[category_col].value_counts().index, palette='viridis')
    plt.title("HALoGEN Dataset: Distribution of Prompt Categories")
    plt.xlabel("Number of Prompts")
    plt.ylabel("Task / Hallucination Category")
    plt.tight_layout()
    plt.savefig("eda_results/category_distribution.png", dpi=300)
    print("Saved category_distribution.png")

    df['word_count'] = df[prompt_col].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(8, 5))
    sns.histplot(df['word_count'], bins=50, kde=True, color='coral')
    plt.title("Distribution of Prompt Lengths in HALoGEN")
    plt.xlabel("Estimated Word Count")
    plt.ylabel("Frequency")
    plt.axvline(df['word_count'].mean(), color='red', linestyle='dashed', linewidth=1,
                label=f"Mean: {df['word_count'].mean():.1f} words")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eda_results/prompt_lengths.png", dpi=300)
    print("Saved prompt_lengths.png")

    summary_df = df[category_col].value_counts().reset_index()
    summary_df.columns = ['Category', 'Count']
    summary_df.to_csv("eda_results/dataset_summary.csv", index=False)
    print("Saved dataset_summary.csv")

    print("\n--- Inspecting Concrete Examples from Different HALoGEN Domains ---")

    sample_domains = df[category_col].unique()[:4]

    for domain in sample_domains:
        example_prompt = df[df[category_col] == domain].iloc[0][prompt_col]

        print(f"\n[Domain: {domain}]")
        print(f"Prompt snippet: {str(example_prompt)[:250]}...")

    print("\nEDA Complete! Check the 'eda_results' folder for your report graphs.")


if __name__ == "__main__":
    run_eda()