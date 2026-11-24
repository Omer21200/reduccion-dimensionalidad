"""Caso Pokémon: PCA, t-SNE y UMAP sobre las 6 estadísticas base (demo en vivo)."""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).parent / "datos" / "Pokemon.csv")
features = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
X = StandardScaler().fit_transform(df[features])
print("Datos:", X.shape, "| legendarios:", int(df["Legendary"].sum()))

# PCA: varianza, autovalores y cargas
pca = PCA().fit(X)
print("\nVarianza explicada:", pca.explained_variance_ratio_.round(3))
print("Autovalores:", pca.explained_variance_.round(2))
print(pd.DataFrame(pca.components_[:2].T, index=features, columns=["PC1", "PC2"]).round(2))

metodos = {
    "PCA": PCA(n_components=2),
    "t-SNE": TSNE(n_components=2, perplexity=30, init="pca", random_state=42),
    "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42),
}

print("\nPreparando UMAP (la primera llamada compila código, no se cuenta en el tiempo)...")
umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X[:200])

leg = df["Legendary"].to_numpy()
iconos = ["Pikachu", "Charizard", "Mewtwo", "Blissey", "Snorlax", "Shuckle"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
print()
for ax, (nombre, modelo) in zip(axes, metodos.items()):
    t0 = time.time()
    Z = modelo.fit_transform(X)
    seg = time.time() - t0
    tw = trustworthiness(X, Z, n_neighbors=10)
    print(f"{nombre}: {seg:.2f} s | vecinos conservados = {tw:.3f}")
    ax.scatter(Z[~leg, 0], Z[~leg, 1], s=8, c="#6B8FB8", alpha=0.6, label="Normal")
    ax.scatter(Z[leg, 0], Z[leg, 1], s=14, c="#D9822B", label="Legendario")
    for n in iconos:
        i = df.index[df["Name"] == n][0]
        ax.annotate(n, Z[i], fontsize=8, weight="bold")
    ax.set_title(f"{nombre} · {seg:.1f} s · vecinos {tw:.2f}")
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend()
plt.tight_layout()
plt.show()
