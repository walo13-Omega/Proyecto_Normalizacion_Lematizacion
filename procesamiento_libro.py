
from pathlib import Path

import spacy
import pandas as pd
from nltk.stem import SnowballStemmer


# ==================================================
# 1. CARGAR EL MODELO DE ESPAÑOL
# ==================================================

print("Cargando modelo de español de spaCy...")

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    raise SystemExit(
        "No se encontró el modelo de español.\n"
        "Ejecuta: python -m spacy download es_core_news_sm"
    )

print("Modelo cargado correctamente.")


# ==================================================
# 2. CARGAR EL LIBRO
# ==================================================

carpeta_proyecto = Path(__file__).parent

ruta_libro = carpeta_proyecto / "rinconete_y_cortadillo.txt"

print("\nCargando Rinconete y Cortadillo...")

if not ruta_libro.exists():
    raise FileNotFoundError(
        "No se encontró rinconete_y_cortadillo.txt. "
        "Comprueba que esté en la misma carpeta que este programa."
    )

texto_original = ruta_libro.read_text(encoding="utf-8-sig").strip()

if not texto_original:
    raise ValueError("El archivo del libro está vacío.")

print(f"Libro cargado correctamente.")
print(f"Cantidad de caracteres: {len(texto_original):,}")


# ==================================================
# 3. TOKENIZACIÓN
# ==================================================

print("\nProcesando el texto con spaCy...")

# Ajustar el límite al tamaño del libro
nlp.max_length = len(texto_original) + 100

doc = nlp(texto_original)

print("\n--- 1. TOKENIZACIÓN ---")

print(f"Total de tokens: {len(doc):,}")

print("\nPrimeros 20 tokens:")

print([token.text for token in doc[:20]])


# ==================================================
# 4. ELIMINACIÓN DE STOP WORDS Y PUNTUACIÓN
# ==================================================

tokens_relevantes = []
palabras_eliminadas = []
puntuacion_eliminada = []

for token in doc:

    if token.is_space:
        continue

    if token.is_stop:
        palabras_eliminadas.append(token.text)
        continue

    if token.is_punct:
        puntuacion_eliminada.append(token.text)
        continue

    if token.text.strip():
        tokens_relevantes.append(token.text)


print("\n--- 2. LIMPIEZA DEL TEXTO ---")

print(f"Tokens originales: {len(doc):,}")
print(f"Stop words eliminadas: {len(palabras_eliminadas):,}")
print(f"Signos de puntuación eliminados: {len(puntuacion_eliminada):,}")
print(f"Tokens conservados: {len(tokens_relevantes):,}")

print("\nEjemplos de stop words eliminadas:")
print(palabras_eliminadas[:15])

print("\nEjemplos de palabras conservadas:")
print(tokens_relevantes[:15])


# ==================================================
# 5. NORMALIZACIÓN Y LEMATIZACIÓN
# ==================================================

tokens_normalizados = []
transformaciones = {}

for token in doc:

    if token.is_stop or token.is_punct or token.is_space:
        continue

    if not token.text.strip():
        continue

    # Convertir a minúsculas y obtener el lema
    lema = token.lemma_.lower().strip()

    if not lema:
        lema = token.text.lower().strip()

    tokens_normalizados.append(lema)

    # Guardar ejemplos de palabras que cambiaron
    palabra_original = token.text.lower()

    if (
        palabra_original != lema
        and token.is_alpha
        and len(palabra_original) > 2
    ):
        transformaciones.setdefault(palabra_original, lema)


print("\n--- 3. NORMALIZACIÓN Y LEMATIZACIÓN ---")

print(f"Tokens normalizados: {len(tokens_normalizados):,}")

print("\nEjemplos de transformaciones:")

for original, lema in list(transformaciones.items())[:15]:
    print(f"{original} -> {lema}")

print("\nPrimeros 20 tokens normalizados:")
print(tokens_normalizados[:20])


# ==================================================
# 6. GUARDAR EL TEXTO NORMALIZADO
# ==================================================

ruta_salida = (
    carpeta_proyecto / "rinconete_y_cortadillo_normalizado.txt"
)

texto_normalizado = " ".join(tokens_normalizados)

ruta_salida.write_text(
    texto_normalizado,
    encoding="utf-8"
)

print("\nTexto normalizado guardado correctamente.")
print(f"Archivo: {ruta_salida.name}")


# ==================================================
# 7. COMPARACIÓN: STEMMING VS LEMATIZACIÓN
# ==================================================

print("\n--- 4. STEMMING VS LEMATIZACIÓN ---")

stemmer = SnowballStemmer("spanish")

data_comparativa = []
palabras_comparadas = set()

for token in doc:

    if token.is_stop or token.is_punct or token.is_space:
        continue

    if not token.is_alpha:
        continue

    palabra = token.text.lower()
    lema = token.lemma_.lower().strip()

    if not lema:
        continue

    # Evitar repetir palabras
    if palabra in palabras_comparadas:
        continue

    # Seleccionar palabras cuya forma original
    # sea diferente de su lema
    if palabra != lema:

        raiz_stem = stemmer.stem(palabra)

        data_comparativa.append({
            "Palabra original": palabra,
            "Stemming": raiz_stem,
            "Lematización": lema,
            "¿Coinciden?": raiz_stem == lema
        })

        palabras_comparadas.add(palabra)

    # Limitar la tabla a 15 ejemplos
    if len(data_comparativa) >= 15:
        break


df = pd.DataFrame(data_comparativa)

if not df.empty:

    print("\nComparación de resultados:")
    print(df.to_string(index=False))

    ruta_comparacion = (
        carpeta_proyecto / "comparacion_stemming_lematizacion.csv"
    )

    df.to_csv(
        ruta_comparacion,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nComparación guardada correctamente.")
    print(f"Archivo: {ruta_comparacion.name}")

else:

    print("No se encontraron palabras adecuadas para comparar.")


# ==================================================
# 8. RESUMEN FINAL
# ==================================================

print("\n--- RESUMEN DEL PROCESAMIENTO ---")

print(f"Caracteres del libro: {len(texto_original):,}")
print(f"Tokens originales: {len(doc):,}")
print(f"Stop words eliminadas: {len(palabras_eliminadas):,}")
print(f"Puntuación eliminada: {len(puntuacion_eliminada):,}")
print(f"Tokens normalizados: {len(tokens_normalizados):,}")

if len(doc) > 0:

    reduccion = (
        1 - len(tokens_normalizados) / len(doc)
    ) * 100

    print(f"Reducción de tokens: {reduccion:.2f}%")

print("\nProceso de normalización y lematización terminado.")
