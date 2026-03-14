from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ABLATION_DIR = PROJECT_ROOT / "data" / "processed" / "ablation"
PLOT_DIR = ABLATION_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_metric(df, metric):

    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"

    plt.figure(figsize=(8,5))

    sns.barplot(
        data=df,
        x="signal_mode",
        y=metric_mean,
        hue="model",
        ci=None
    )

    # add error bars
    for i,row in df.iterrows():
        plt.errorbar(
            x=i % len(df.signal_mode.unique()),
            y=row[metric_mean],
            yerr=row[metric_std],
            fmt='none',
            capsize=4,
            color='black'
        )

    plt.title(f"{metric.upper()} comparison across signal types")
    plt.ylabel(metric.upper())
    plt.xlabel("Signal mode")

    plt.ylim(0,1)

    plt.legend(title="Model")

    outfile = PLOT_DIR / f"{metric}_comparison.png"

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)

    plt.close()

    print(f"Saved {outfile}")


def main():

    summary_path = ABLATION_DIR / "interaction_ablation_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            "Run ablation first: python scripts/run_interaction_ablation.py"
        )

    df = pd.read_csv(summary_path)

    # nicer model names
    df["model"] = df["model"].replace({
        "federated_pytorch":"Federated MLP",
        "centralized_logreg":"Logistic Regression"
    })

    plot_metric(df,"accuracy")
    plot_metric(df,"f1")
    plot_metric(df,"auroc")

    print("\nPlots saved in:")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()