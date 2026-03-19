import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, : self.n_components].T
        self.eigenvalues = eigenvalues[: self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        return X_reduced @ self.components + self.mean


def demo_synthetic():
    print("=" * 60)
    print("PCA on synthetic 3D data")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 500

    t = np.random.uniform(0, 2 * np.pi, n_samples)
    x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
    x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
    x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)

    X = np.column_stack([x1, x2, x3])

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Reduced shape:  {X_reduced.shape}")
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")

    X_reconstructed = pca.inverse_transform(X_reduced)
    mse = np.mean((X - X_reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    print()


def demo_mnist():
    print("=" * 60)
    print("PCA on MNIST digits")
    print("=" * 60)

    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data[:5000].astype(float)
    y = mnist.target[:5000].astype(int)

    pca_50 = PCA(n_components=50)
    X_pca50 = pca_50.fit_transform(X)
    print(f"50 components capture {sum(pca_50.explained_variance_ratio_):.2%} of variance")

    pca_2d = PCA(n_components=2)
    X_pca2d = pca_2d.fit_transform(X)
    print(f"2 components capture {sum(pca_2d.explained_variance_ratio_):.2%} of variance")

    for k in [10, 50, 200]:
        pca_k = PCA(n_components=k)
        X_k = pca_k.fit_transform(X)
        X_rec = pca_k.inverse_transform(X_k)
        mse = np.mean((X - X_rec) ** 2)
        var = sum(pca_k.explained_variance_ratio_)
        print(f"k={k:>3d}  variance={var:.4f}  reconstruction_mse={mse:.2f}")

    print()
    return X, y, X_pca2d


def demo_sklearn_comparison(X, X_ours):
    print("=" * 60)
    print("Comparison: our PCA vs sklearn PCA")
    print("=" * 60)

    from sklearn.decomposition import PCA as SklearnPCA

    sklearn_pca = SklearnPCA(n_components=2)
    X_sklearn = sklearn_pca.fit_transform(X)

    pca_ours = PCA(n_components=2)
    pca_ours.fit(X)

    print(f"Our explained variance:     {pca_ours.explained_variance_ratio_}")
    print(f"Sklearn explained variance: {sklearn_pca.explained_variance_ratio_}")

    diff = np.abs(np.abs(X_ours) - np.abs(X_sklearn))
    print(f"Max absolute difference (sign-invariant): {diff.max():.10f}")
    print()


def demo_tsne(X, y):
    print("=" * 60)
    print("t-SNE on MNIST (5000 samples)")
    print("=" * 60)

    from sklearn.manifold import TSNE

    pca_pre = PCA(n_components=50)
    X_pca = pca_pre.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    print(f"t-SNE output shape: {X_tsne.shape}")
    print(f"t-SNE x range: [{X_tsne[:, 0].min():.1f}, {X_tsne[:, 0].max():.1f}]")
    print(f"t-SNE y range: [{X_tsne[:, 1].min():.1f}, {X_tsne[:, 1].max():.1f}]")
    print()


def demo_umap(X, y):
    print("=" * 60)
    print("UMAP on MNIST (5000 samples)")
    print("=" * 60)

    try:
        from umap import UMAP

        pca_pre = PCA(n_components=50)
        X_pca = pca_pre.fit_transform(X)

        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X_pca)
        print(f"UMAP output shape: {X_umap.shape}")
        print(f"UMAP x range: [{X_umap[:, 0].min():.1f}, {X_umap[:, 0].max():.1f}]")
        print(f"UMAP y range: [{X_umap[:, 1].min():.1f}, {X_umap[:, 1].max():.1f}]")
    except ImportError:
        print("Install umap-learn to run this demo: pip install umap-learn")

    print()


def demo_pca_preprocessing(X, y):
    print("=" * 60)
    print("PCA as preprocessing for logistic regression")
    print("=" * 60)

    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for k in [10, 30, 50, 100, 200, 784]:
        if k < X_train.shape[1]:
            pca_k = SklearnPCA(n_components=k)
            X_tr = pca_k.fit_transform(X_train)
            X_te = pca_k.transform(X_test)
            var_captured = sum(pca_k.explained_variance_ratio_)
        else:
            X_tr = X_train
            X_te = X_test
            var_captured = 1.0

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_train)
        acc = accuracy_score(y_test, clf.predict(X_te))
        print(f"k={k:>3d}  accuracy={acc:.4f}  variance={var_captured:.4f}")

    print()


if __name__ == "__main__":
    demo_synthetic()

    X, y, X_pca2d = demo_mnist()
    demo_sklearn_comparison(X, X_pca2d)
    demo_tsne(X, y)
    demo_umap(X, y)
    demo_pca_preprocessing(X, y)
