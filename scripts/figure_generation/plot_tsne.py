import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Plot t-SNE panels from exported SFibAI features.")
    parser.add_argument("--features", required=True, help="NPZ file created by extract_features.py.")
    parser.add_argument("--save_dir", default=str(repo_root / "artifacts" / "figures" / "tsne"))
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning_rate", default="auto")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--init", default="pca", choices=["pca", "random"])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--pca_components", type=int, default=50,
                        help="Set to 0 to disable PCA preprocessing.")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def _prepare_features(features, pca_components, random_state):
    if pca_components <= 0 or pca_components >= features.shape[1]:
        return features
    n_components = min(pca_components, features.shape[0], features.shape[1])
    return PCA(n_components=n_components, random_state=random_state).fit_transform(features)


def _scatter(embedding, values, title, colorbar_label, save_path, dpi, cmap="viridis"):
    plt.figure(figsize=(7.2, 6.4))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values,
        s=12,
        cmap=cmap,
        alpha=0.82,
        linewidths=0,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorbar_label)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    archive = np.load(args.features, allow_pickle=True)
    features = archive["features"]
    scores = archive["scores"]
    grades = archive["grades"]

    prepared = _prepare_features(features, args.pca_components, args.random_state)
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": args.perplexity,
        "learning_rate": args.learning_rate,
        "init": args.init,
        "random_state": args.random_state,
    }
    try:
        tsne = TSNE(max_iter=args.max_iter, **tsne_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=args.max_iter, **tsne_kwargs)
    embedding = tsne.fit_transform(prepared)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
            "score": scores,
            "clinical_grade": grades,
        }
    ).to_csv(save_dir / "tsne_coordinates.csv", index=False)

    _scatter(
        embedding,
        scores,
        "t-SNE of Stage 4 Features by 36-Level Score",
        "Fibrosis score",
        save_dir / "tsne_36_level.pdf",
        args.dpi,
        cmap="viridis",
    )
    _scatter(
        embedding,
        grades,
        "t-SNE of Stage 4 Features by Clinical Grade",
        "Clinical grade",
        save_dir / "tsne_4_grade.pdf",
        args.dpi,
        cmap="plasma",
    )

    with open(save_dir / "tsne_parameters.txt", "w", encoding="utf-8") as handle:
        handle.write(f"features={args.features}\n")
        handle.write(f"perplexity={args.perplexity}\n")
        handle.write(f"learning_rate={args.learning_rate}\n")
        handle.write(f"max_iter={args.max_iter}\n")
        handle.write(f"init={args.init}\n")
        handle.write(f"random_state={args.random_state}\n")
        handle.write(f"pca_components={args.pca_components}\n")
        handle.write(f"n_samples={features.shape[0]}\n")
        handle.write(f"feature_dim={features.shape[1]}\n")
    print(f"Saved t-SNE outputs to: {save_dir}")


if __name__ == "__main__":
    main()
