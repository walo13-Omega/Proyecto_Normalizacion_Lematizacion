
from pathlib import Path

import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.sparse import save_npz


# =========================================================
# 1. CONFIGURACIÓN Y CARGA DEL LIBRO
# =========================================================

carpeta_proyecto = Path(__file__).parent
ruta_libro = carpeta_proyecto / "rinconete_y_cortadillo.txt"

print("Cargando modelo de español de spaCy...")

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    raise SystemExit(
        "No se encontró el modelo de español. "
        "Ejecuta: python -m spacy download es_core_news_sm"
    )

if not ruta_libro.exists():
    raise FileNotFoundError(
        "No se encontró rinconete_y_cortadillo.txt."
    )

texto = ruta_libro.read_text(encoding="utf-8-sig").strip()

if not texto:
    raise ValueError("El archivo del libro está vacío.")

nlp.max_length = len(texto) + 100

print(f"Libro cargado: {len(texto):,} caracteres.")


# =========================================================
# 2. PROCESAMIENTO Y LEMATIZACIÓN POR ORACIÓN
# =========================================================

print("Procesando el texto con spaCy...")

doc = nlp(texto)
corpus_lematizado = []

for oracion in doc.sents:

    lemas_oracion = [
        token.lemma_.lower().strip()
        for token in oracion
        if not token.is_punct
        and not token.is_space
        and not token.is_stop
        and token.lemma_.strip()
        and token.is_alpha
    ]

    if lemas_oracion:
        corpus_lematizado.append(" ".join(lemas_oracion))

print(f"Total de oraciones procesadas: {len(corpus_lematizado):,}")

if len(corpus_lematizado) < 3:
    raise ValueError(
        "Se necesitan al menos tres oraciones para aplicar PCA en 3D."
    )

# Guardar el corpus para documentar esta etapa.
ruta_corpus = carpeta_proyecto / "corpus_lematizado_oraciones.txt"

ruta_corpus.write_text(
    "\n".join(corpus_lematizado),
    encoding="utf-8"
)


# =========================================================
# 3. BAG OF WORDS (BoW)
# =========================================================

print("\nAplicando Bag of Words...")

bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)

vocabulario_bow = bow_vectorizer.get_feature_names_out()

print("BoW generado correctamente.")
print(f"Documentos (oraciones): {X_bow.shape[0]:,}")
print(f"Palabras del vocabulario: {X_bow.shape[1]:,}")
print(f"Dimensiones de la matriz BoW: {X_bow.shape}")


# =========================================================
# 4. TF-IDF
# =========================================================

print("\nAplicando TF-IDF...")

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)

vocabulario_tfidf = tfidf_vectorizer.get_feature_names_out()

print("TF-IDF generado correctamente.")
print(f"Dimensiones de la matriz TF-IDF: {X_tfidf.shape}")


# =========================================================
# 5. GUARDAR LAS MATRICES VECTORIALES
# =========================================================

save_npz(carpeta_proyecto / "matriz_bow.npz", X_bow)
save_npz(carpeta_proyecto / "matriz_tfidf.npz", X_tfidf)

print("\nMatrices guardadas en formato NPZ.")


# =========================================================
# 6. COMPARACIÓN DE LAS REPRESENTACIONES
# =========================================================

frecuencias = np.asarray(X_bow.sum(axis=0)).ravel()
pesos_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()

df_resultados = pd.DataFrame({
    "Palabra": vocabulario_bow,
    "Frecuencia_BoW": frecuencias,
    "Peso_promedio_TF_IDF": pesos_tfidf
})

df_resultados = df_resultados.sort_values(
    by="Frecuencia_BoW",
    ascending=False
)

ruta_resultados = carpeta_proyecto / "resultados_vectorizacion.csv"

df_resultados.to_csv(
    ruta_resultados,
    index=False,
    encoding="utf-8-sig"
)

print("\nPrimeras 15 palabras del análisis:")
print(df_resultados.head(15).to_string(index=False))


# =========================================================
# 7. REDUCCIÓN DE DIMENSIONALIDAD CON PCA
# =========================================================

def graficar_palabras_3d(ax, matriz, vocabulario, titulo):

    # Transponer: cada fila representa una palabra
    matriz_palabras = matriz.T

    # PCA reduce las características a tres dimensiones
    pca = PCA(n_components=3)
    coordenadas = pca.fit_transform(matriz_palabras.toarray())

    x = coordenadas[:, 0]
    y = coordenadas[:, 1]
    z = coordenadas[:, 2]

    ax.scatter(
        x, y, z,
        s=65,
        edgecolors="black",
        alpha=0.8,
        depthshade=True
    )

    for i, palabra in enumerate(vocabulario):
        ax.text(
            x[i], y[i], z[i] + 0.02,
            palabra,
            fontsize=8
        )

    ax.set_title(titulo)
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.set_zlabel("Componente principal 3")

    varianza = pca.explained_variance_ratio_.sum() * 100
    print(f"{titulo}: varianza explicada = {varianza:.2f}%")


# Seleccionar las 40 palabras más frecuentes para que
# las etiquetas de la gráfica sean legibles.
cantidad_palabras = min(20, X_bow.shape[1])

if cantidad_palabras < 3:
    raise ValueError(
        "El vocabulario es demasiado pequeño para generar la gráfica."
    )

indices = np.argsort(frecuencias)[-cantidad_palabras:][::-1]

palabras_seleccionadas = vocabulario_bow[indices]

bow_seleccionado = X_bow[:, indices]
tfidf_seleccionado = X_tfidf[:, indices]

fig = plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(121, projection="3d")
graficar_palabras_3d(
    ax1,
    bow_seleccionado,
    palabras_seleccionadas,
    "Espacio vectorial BoW"
)

ax2 = fig.add_subplot(122, projection="3d")
graficar_palabras_3d(
    ax2,
    tfidf_seleccionado,
    palabras_seleccionadas,
    "Espacio vectorial TF-IDF"
)

plt.tight_layout()

ruta_grafica = carpeta_proyecto / "comparacion_vectorial_3d.png"

plt.savefig(
    ruta_grafica,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nGráfica guardada: {ruta_grafica.name}")

plt.show()


# =========================================================
# 8. RESUMEN FINAL
# =========================================================

print("\n========== RESUMEN DE VECTORIZACIÓN ==========")
print(f"Oraciones procesadas: {len(corpus_lematizado):,}")
print(f"Palabras únicas: {len(vocabulario_bow):,}")
print(f"Matriz BoW: {X_bow.shape}")
print(f"Matriz TF-IDF: {X_tfidf.shape}")
print(f"Palabras representadas en PCA: {cantidad_palabras}")

print("\nArchivos generados:")
print("- corpus_lematizado_oraciones.txt")
print("- matriz_bow.npz")
print("- matriz_tfidf.npz")
print("- resultados_vectorizacion.csv")
print("- comparacion_vectorial_3d.png")

print("\nProceso de vectorización terminado correctamente.")
