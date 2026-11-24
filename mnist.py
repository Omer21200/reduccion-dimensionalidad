"""Caso MNIST: PCA, t-SNE y UMAP sobre 5.000 dígitos escritos a mano, y K-means al final."""
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score

# Datos: se descargan la primera vez (11 MB), igual que keras.datasets.mnist
ruta = Path(__file__).parent / "datos" / "mnist.npz"
if not ruta.exists():
    print("Descargando MNIST (11 MB)...")
    ruta.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", ruta)
with np.load(ruta) as f:
    x_train, y_train = f["x_train"], f["y_train"]

# Muestra de 5.000 imágenes, aplanadas a 784 píxeles en [0, 1]
N = 5000
idx = np.random.default_rng(42).choice(len(x_train), N, replace=False)
X = x_train[idx].reshape(N, -1) / 255.0
y = y_train[idx]
print("Datos:", X.shape)

# PCA: ¿cuánta varianza se conserva?
var_acum = np.cumsum(PCA().fit(X).explained_variance_ratio_)
print(f"Varianza con 2 componentes: {var_acum[1]:.3f} | con 50: {var_acum[49]:.3f} | "
      f"90 % con {np.argmax(var_acum >= 0.90) + 1} componentes")

# Buena práctica: PCA a 50 antes de t-SNE y UMAP
X50 = PCA(n_components=50, random_state=42).fit_transform(X)

print("\nPreparando UMAP (la primera llamada compila código, tarda 1-2 min y no se cuenta)...")
umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X50)

metodos = {
    "PCA": (PCA(n_components=2, random_state=42), X),
    "t-SNE": (TSNE(n_components=2, perplexity=30, init="pca", random_state=42), X50),
    "UMAP": (umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42), X50),
}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
proyecciones = {}
print()
for ax, (nombre, (modelo, entrada)) in zip(axes, metodos.items()):
    t0 = time.time()
    Z = modelo.fit_transform(entrada)
    seg = time.time() - t0
    tw = trustworthiness(X, Z, n_neighbors=10)
    proyecciones[nombre] = Z
    print(f"{nombre}: {seg:.2f} s | vecinos conservados = {tw:.3f}")
    ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=3)
    for d in range(10):
        cx, cy = np.median(Z[y == d], axis=0)
        ax.text(cx, cy, str(d), fontsize=14, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle", fc="white", alpha=0.8))
    ax.set_title(f"{nombre} · {seg:.1f} s · vecinos {tw:.2f}")
    ax.set_xticks([]); ax.set_yticks([])


# K-means con 10 grupos: ¿coinciden con los dígitos reales? (ARI: 1 = perfecto)
def ari(Z):
    return adjusted_rand_score(y, KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(Z))


print("\nK-means (ARI)")
print(f"  784 píxeles: {ari(X):.2f}")
print(f"  PCA 50:      {ari(X50):.2f}")
print(f"  PCA 2D:      {ari(proyecciones['PCA']):.2f}")
print(f"  UMAP 2D:     {ari(proyecciones['UMAP']):.2f}")
print("  (t-SNE no se usa para K-means: sus distancias no tienen significado)")

plt.tight_layout()
plt.show()
