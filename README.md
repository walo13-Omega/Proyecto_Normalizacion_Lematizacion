

\# Proyecto de Normalización y Lematización de Texto



\## Descripción



Este proyecto académico aplica técnicas de Procesamiento de Lenguaje Natural (PLN) sobre la obra literaria \*Rinconete y Cortadillo\*, de Miguel de Cervantes.



El objetivo es realizar un proceso de limpieza y transformación del texto mediante tokenización, eliminación de palabras vacías, eliminación de signos de puntuación, normalización y lematización.



También se realiza una comparación entre los resultados de stemming y lematización utilizando herramientas de Python.



\## Obra utilizada



\- \*\*Título:\*\* Rinconete y Cortadillo

\- \*\*Autor:\*\* Miguel de Cervantes

\- \*\*Idioma:\*\* Español

\- \*\*Fuente:\*\* Wikisource

\- \*\*Enlace:\*\* https://es.wikisource.org/wiki/Rinconete\_y\_Cortadillo\_(1825)



\## Tecnologías utilizadas



\- Python

\- spaCy

\- Modelo de español `es\_core\_news\_sm`

\- NLTK

\- pandas



\## Estructura del proyecto



```text

Proyecto\_Normalizacion\_Lematizacion/

│

├── descargar\_libro.py

├── rinconete\_y\_cortadillo.txt

├── procesamiento\_libro.py

├── rinconete\_y\_cortadillo\_normalizado.txt

├── comparacion\_stemming\_lematizacion.csv

├── requirements.txt

├── README.md

└── .gitignore

```



\## Instalación



Crear un entorno virtual:



```powershell

python -m venv venv

```



Activar el entorno:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Instalar las dependencias:



```powershell

pip install -r requirements.txt

```



Instalar el modelo de español:



```powershell

python -m spacy download es\_core\_news\_sm

```



\## Ejecución



Para descargar el libro:



```powershell

python descargar\_libro.py

```



Para realizar el procesamiento:



```powershell

python procesamiento\_libro.py

```



\## Procesamiento aplicado



1\. \*\*Tokenización:\*\* separación del texto en unidades individuales llamadas tokens.

2\. \*\*Eliminación de stop words:\*\* eliminación de palabras frecuentes que aportan poca información por sí solas.

3\. \*\*Eliminación de puntuación:\*\* retiro de signos de puntuación.

4\. \*\*Normalización:\*\* conversión de las palabras a minúsculas.

5\. \*\*Lematización:\*\* transformación de las palabras a su forma base mediante spaCy.

6\. \*\*Stemming:\*\* reducción de palabras mediante SnowballStemmer de NLTK.



\## Diferencia entre stemming y lematización



El stemming reduce las palabras mediante reglas que eliminan o modifican terminaciones. En cambio, la lematización utiliza información lingüística para identificar la forma base de una palabra.



Por esta razón, ambos métodos pueden producir resultados diferentes.



\## Archivos generados



\- `rinconete\_y\_cortadillo\_normalizado.txt`: contiene el texto después de aplicar la limpieza y lematización.

\- `comparacion\_stemming\_lematizacion.csv`: contiene ejemplos de comparación entre ambas técnicas.



