
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from pathlib import Path


URL = "https://es.wikisource.org/wiki/Rinconete_y_Cortadillo_(1825)"


class ExtractorTexto(HTMLParser):
    def __init__(self):
        super().__init__()
        self.dentro_contenido = False
        self.profundidad = 0
        self.ignorar = 0
        self.fragmentos = []

    def handle_starttag(self, tag, attrs):
        atributos = dict(attrs)

        if not self.dentro_contenido:
            if tag == "div" and atributos.get("id") == "mw-content-text":
                self.dentro_contenido = True
                self.profundidad = 1
            return

        self.profundidad += 1

        if tag in ("script", "style"):
            self.ignorar += 1

    def handle_endtag(self, tag):
        if not self.dentro_contenido:
            return

        if tag in ("script", "style") and self.ignorar:
            self.ignorar -= 1

        self.profundidad -= 1

        if self.profundidad <= 0:
            self.dentro_contenido = False

    def handle_data(self, data):
        if self.dentro_contenido and not self.ignorar:
            texto = data.strip()
            if texto:
                self.fragmentos.append(texto)


print("Descargando Rinconete y Cortadillo...")

solicitud = Request(
    URL,
    headers={"User-Agent": "Proyecto educativo de PLN"}
)

with urlopen(solicitud, timeout=30) as respuesta:
    html = respuesta.read().decode("utf-8")

extractor = ExtractorTexto()
extractor.feed(html)

texto = "\n".join(extractor.fragmentos)
# Eliminar encabezados y elementos de navegación de Wikisource
inicio = texto.find("En la venta del Molinillo")

if inicio == -1:
    raise ValueError("No se encontró el inicio del texto de la novela.")

texto = texto[inicio:]

# Eliminar el pie de página de Wikisource
fin = texto.find("Obtenido de «")

if fin != -1:
    texto = texto[:fin]

texto = texto.strip()

if len(texto) < 1000:
    raise ValueError("No se pudo extraer suficiente texto de la página.")

ruta = Path(__file__).parent / "rinconete_y_cortadillo.txt"
ruta.write_text(texto, encoding="utf-8")

print("Libro guardado correctamente.")
print(f"Caracteres descargados: {len(texto):,}")
print(f"Archivo: {ruta.name}")
