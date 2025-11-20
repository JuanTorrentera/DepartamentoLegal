import logging
import os

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from pypdf import PdfReader

# --- 1. CONFIGURACIÓN ---
# Configura el logging para ver los mensajes de información
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración del modelo de embedding y el LLM
# Usaremos un modelo de embedding local y Groq como LLM
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama3-8b-8192"  # Modelo rápido y capaz de Groq

# Ruta al documento PDF
PDF_PATH = "LIElec_090321.pdf"

# Variable global para el query engine
query_engine = None

# --- 2. APLICACIÓN FASTAPI ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")


# --- 3. LÓGICA DE CARGA E INDEXACIÓN ---
def load_pdf_documents(pdf_path: str):
    """Carga un PDF y lo divide en objetos Document por página."""
    if not os.path.exists(pdf_path):
        logger.error(f"El archivo PDF no se encuentra en la ruta: {pdf_path}")
        raise FileNotFoundError(
            f"El archivo PDF no se encuentra en la ruta: {pdf_path}"
        )

    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(
            Document(
                text=text,
                metadata={"page": i + 1, "source": os.path.basename(pdf_path)},
            )
        )

    logger.info(f"Se cargaron {len(docs)} páginas del documento.")
    return docs


@app.on_event("startup")
def startup_event():
    """
    Función que se ejecuta al iniciar la aplicación.
    Carga el LLM, el modelo de embedding, el PDF y crea el índice.
    """
    global query_engine
    logger.info("Iniciando la aplicación...")

    # Verificar si la API key de Groq está disponible
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("La variable de entorno GROQ_API_KEY no está configurada.")
        raise ValueError("Debes configurar la variable de entorno GROQ_API_KEY.")

    # Configurar el LLM con Groq
    Settings.llm = Groq(model=LLM_MODEL_NAME, api_key=groq_api_key)
    logger.info(f"LLM configurado con el modelo: {LLM_MODEL_NAME}")

    # Configurar el modelo de embedding
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    logger.info(f"Modelo de embedding configurado: {EMBEDDING_MODEL_NAME}")

    # Cargar los documentos del PDF
    documents = load_pdf_documents(PDF_PATH)

    # Crear el índice en memoria
    logger.info("Creando el índice vectorial en memoria...")
    index = VectorStoreIndex.from_documents(documents)

    # Crear el query engine
    query_engine = index.as_query_engine(similarity_top_k=3)
    logger.info("¡El sistema está listo para recibir preguntas!")


# --- 4. RUTAS DE LA API ---
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Sirve la interfaz de chat en HTML."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query", response_class=HTMLResponse)
async def handle_query(request: Request, question: str = Form(...)):
    """Maneja las preguntas del usuario y devuelve la respuesta."""
    if not query_engine:
        return HTMLResponse(
            "<p>El motor de búsqueda no está inicializado. Revisa los logs del servidor.</p>",
            status_code=500,
        )

    logger.info(f"Recibida pregunta: {question}")

    if question.lower().strip() == "resumen":
        # Usar un modo específico para resumen si es necesario
        summarizer = query_engine.index.as_query_engine(response_mode="tree_summarize")
        response = await summarizer.aquery(
            "Genera un resumen detallado y estructurado del documento."
        )
    else:
        response = await query_engine.aquery(question)

    response_text = str(response)

    # Formatear la respuesta como HTML
    formatted_response = f"""
    <div class='p-4 mb-4 text-sm text-blue-800 rounded-lg bg-blue-50' role='alert'>
        <span class='font-medium'>Respuesta:</span> {response_text}
    </div>
    """

    return HTMLResponse(content=formatted_response)


# --- 5. EJECUCIÓN (para desarrollo local) ---
if __name__ == "__main__":
    import uvicorn

    # Imprimir advertencia sobre la API Key
    if not os.environ.get("GROQ_API_KEY"):
        print("\n" + "=" * 50)
        print("ADVERTENCIA: La variable de entorno GROQ_API_KEY no está configurada.")
        print("El programa intentará ejecutarse, pero fallará al iniciar.")
        print("Crea un archivo .env y añade: GROQ_API_KEY='TU_API_KEY'")
        print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
