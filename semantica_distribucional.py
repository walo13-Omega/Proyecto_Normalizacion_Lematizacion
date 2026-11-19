
from pathlib import Path

import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.decomposition import PCA


# =========================================================
# 1. CONFIGURACIÓN Y CARGA DEL LIBRO
# =========================================================

carpeta = Path(__file__).parent
ruta_libro = carpeta / "rinconete_y_cortadillo.txt"

print("Cargando modelo de español...")

nlp = spacy.load("es_core_news_sm")

if not ruta_libro.exists():
    raise FileNotFoundError("No se encontró el archivo del libro.")

texto = ruta_libro.read_text(encoding="utf-8-sig").strip()

if not texto:
    raise ValueError("El archivo del libro está vacío.")

nlp.max_length = len(texto) + 100

print(f"Libro cargado: {len(texto):,} caracteres.")


# =========================================================
# 2. LIMPIEZA Y PREPARACIÓN DEL CORPUS
# =========================================================

print("\nProcesando texto con spaCy...")

doc = nlp(texto)
sentences = []

for sent in doc.sents:

    tokens = [
        token.lemma_.lower().strip()
        for token in sent
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
        and token.lemma_.strip()
    ]

    if len(tokens) > 1:
        sentences.append(tokens)

print(f"Total de oraciones procesadas: {len(sentences)}")

if len(sentences) < 3:
    raise ValueError("No hay suficientes oraciones para entrenar Word2Vec.")

print("\nEjemplo de oración procesada:")
print(sentences[0])

# Guardar el corpus utilizado para el entrenamiento.
ruta_corpus = carpeta / "corpus_semantico_lematizado.txt"

ruta_corpus.write_text(
    "\n".join(" ".join(oracion) for oracion in sentences),
    encoding="utf-8"
)


# =========================================================
# 3. ENTRENAMIENTO DEL MODELO WORD2VEC
# =========================================================

print("\nEntrenando modelo Word2Vec...")

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=2,
    workers=1,
    sg=1,
    epochs=100,
    seed=42
)

print("Entrenamiento terminado.")

print(f"Palabras en el vocabulario: {len(model.wv.index_to_key)}")
print(f"Dimensiones de cada embedding: {model.vector_size}")

if len(model.wv.index_to_key) < 3:
    raise ValueError("El vocabulario es demasiado pequeño para PCA 3D.")

# Guardar el modelo entrenado.
ruta_modelo = carpeta / "modelo_word2vec.model"
model.save(str(ruta_modelo))


# =========================================================
# 4. EXPLORACIÓN DE PALABRAS SIMILARES
# =========================================================

print("\n========== PALABRAS SEMÁNTICAMENTE SIMILARES ==========")

palabras_consulta = [
    "monipodio",
    "rincón",
    "cortadillo",
    "señor",
    "casa",
    "oficio"
]

resultados_similares = []

for palabra in palabras_consulta:

    if palabra not in model.wv:
        print(f"\n'{palabra}' no está en el vocabulario.")
        continue

    similares = model.wv.most_similar(
        palabra,
        topn=min(5, len(model.wv) - 1)
    )

    print(f"\nPalabras cercanas a '{palabra}':")

    for palabra_similar, puntuacion in similares:

        print(f"  {palabra_similar}: {puntuacion:.4f}")

        resultados_similares.append({
            "Palabra_consultada": palabra,
            "Palabra_similar": palabra_similar,
            "Similitud": puntuacion
        })

df_similares = pd.DataFrame(resultados_similares)

ruta_similares = carpeta / "palabras_similares_word2vec.csv"

df_similares.to_csv(
    ruta_similares,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 5. ARITMÉTICA SEMÁNTICA
# =========================================================

print("\n========== EXPERIMENTO DE ARITMÉTICA SEMÁNTICA ==========")

# Operación vectorial:
# monipodio + rincón - señor
#
# Word2Vec permite realizar operaciones matemáticas
# con los embeddings y buscar palabras cercanas al resultado.

positivas = ["monipodio", "rincón"]
negativas = ["señor"]

if all(p in model.wv for p in positivas + negativas):

    resultado = model.wv.most_similar(
        positive=positivas,
        negative=negativas,
        topn=5
    )

    print("Operación: monipodio + rincón - señor")

    for palabra, puntuacion in resultado:
        print(f"  {palabra}: {puntuacion:.4f}")

    texto_aritmetica = [
        "EXPERIMENTO DE ARITMETICA SEMANTICA",
        "Operacion: monipodio + rincon - senor",
        "",
        "Resultados:"
    ]

    texto_aritmetica.extend(
        f"{palabra}: {puntuacion:.4f}"
        for palabra, puntuacion in resultado
    )

else:
    print("No están todas las palabras necesarias en el vocabulario.")
    texto_aritmetica = [
        "No se pudo realizar la operación.",
        "Faltan palabras en el vocabulario."
    ]

(carpeta / "aritmetica_semantica.txt").write_text(
    "\n".join(texto_aritmetica),
    encoding="utf-8"
)


# =========================================================
# 6. EXPORTAR LOS EMBEDDINGS
# =========================================================

print("\nExportando embeddings...")

vocabulario = model.wv.index_to_key

vectores = np.vstack([
    model.wv[palabra] for palabra in vocabulario
])

df_embeddings = pd.DataFrame(
    vectores,
    columns=[
        f"dimension_{i + 1}"
        for i in range(model.vector_size)
    ]
)

df_embeddings.insert(0, "Palabra", vocabulario)

ruta_embeddings = carpeta / "embeddings_word2vec.csv"

df_embeddings.to_csv(
    ruta_embeddings,
    index=False,
    encoding="utf-8-sig"
)

print(f"Embeddings exportados: {df_embeddings.shape}")


# =========================================================
# 7. VISUALIZACIÓN 3D DEL ESPACIO SEMÁNTICO
# =========================================================

print("\nGenerando espacio semántico 3D...")

pca = PCA(n_components=3)

vectores_3d = pca.fit_transform(vectores)

# Obtener frecuencias para identificar palabras destacadas.
frecuencias = {
    palabra: model.wv.get_vecattr(palabra, "count")
    for palabra in vocabulario
}

# Seleccionar las 30 palabras más frecuentes para etiquetar.
palabras_destacadas = sorted(
    vocabulario,
    key=lambda palabra: frecuencias[palabra],
    reverse=True
)[:30]

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    vectores_3d[:, 0],
    vectores_3d[:, 1],
    vectores_3d[:, 2],
    s=25,
    alpha=0.55
)

for palabra in palabras_destacadas:

    indice = vocabulario.index(palabra)

    ax.text(
        vectores_3d[indice, 0],
        vectores_3d[indice, 1],
        vectores_3d[indice, 2],
        palabra,
        fontsize=8
    )

ax.set_title(
    "Espacio semántico de Rinconete y Cortadillo - Word2Vec"
)

ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
ax.set_zlabel("Componente principal 3")

varianza = pca.explained_variance_ratio_.sum() * 100

print(f"Varianza explicada por PCA: {varianza:.2f}%")

fig.tight_layout()

ruta_grafica = carpeta / "espacio_semantico_word2vec_3d.png"

fig.savefig(
    ruta_grafica,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# 8. HEATMAP DE LOS EMBEDDINGS
# =========================================================

print("\nGenerando heatmap de embeddings...")

cantidad = min(30, len(vocabulario))

palabras_heatmap = vocabulario[:cantidad]

matriz_heatmap = np.vstack([
    model.wv[palabra]
    for palabra in palabras_heatmap
])

fig, ax = plt.subplots(figsize=(16, 9))

imagen = ax.imshow(
    matriz_heatmap,
    cmap="RdBu_r",
    aspect="auto"
)

ax.set_yticks(range(cantidad))
ax.set_yticklabels(palabras_heatmap, fontsize=9)

ax.set_xlabel("Dimensiones del embedding")
ax.set_ylabel("Palabras")

ax.set_title(
    "Embeddings Word2Vec - Rinconete y Cortadillo"
)

fig.colorbar(
    imagen,
    ax=ax,
    label="Valor del vector"
)

fig.tight_layout()

ruta_heatmap = carpeta / "heatmap_embeddings_word2vec.png"

fig.savefig(
    ruta_heatmap,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# 9. RESUMEN FINAL
# =========================================================

print("\n========== RESUMEN FINAL ==========")

print(f"Oraciones procesadas: {len(sentences)}")
print(f"Palabras en vocabulario: {len(vocabulario)}")
print(f"Dimensiones por palabra: {model.vector_size}")
print(f"Varianza explicada por PCA: {varianza:.2f}%")

print("\nArchivos generados:")

print("- corpus_semantico_lematizado.txt")
print("- modelo_word2vec.model")
print("- embeddings_word2vec.csv")
print("- palabras_similares_word2vec.csv")
print("- aritmetica_semantica.txt")
print("- espacio_semantico_word2vec_3d.png")
print("- heatmap_embeddings_word2vec.png")

print("\nProceso de semántica distribucional terminado.")
