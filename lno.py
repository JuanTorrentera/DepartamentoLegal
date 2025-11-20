import os
import sys

# --- 1. IMPORTACIONES ---
# Importaciones de componentes base (llama_index.core)
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.response_synthesizers import ResponseMode 

# ✅ CORRECCIÓN FINAL: Importación de Ollama desde el módulo de integraciones (Requiere 'llama-index-llms-ollama')
from llama_index.llms import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- 2. CONFIGURACIÓN INICIAL Y OPTIMIZACIÓN ---
# ⚠️ RUTA DE ARCHIVO: Asegúrate de que esta ruta sea correcta en tu sistema
PDF_PATH = r"C:\Users\juanm\OneDrive\Escritorio\Departamento legal\LIElec_090321.pdf"
CHROMA_DB_PATH = "./chroma_db_llama" 

# 🟢 MODELO ACTUALIZADO: Usando Gemma 3:4b (Asegúrate de que esté descargado en Ollama)
LLM_MODEL_NAME = "gemma3:4b" 

# 🧠 EMBEDDING MODEL: Modelo eficiente para generar los vectores
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- CONFIGURACIÓN DE LLAMAINDEX (OPTIMIZADA) ---
try:
    # Inicialización del LLM. request_timeout es alto para resúmenes largos.
    Settings.llm = Ollama(model=LLM_MODEL_NAME, request_timeout=360.0) 
except Exception as e:
    print(f"⚠️ ADVERTENCIA: No se pudo conectar a Ollama o cargar el modelo '{LLM_MODEL_NAME}'.")
    print("Asegúrese de que Ollama esté corriendo y el modelo esté descargado (ollama pull gemma3:4b).")
    print(f"Detalle: {e}")
    sys.exit(1) # Salir si el LLM esencial no está disponible
    
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
# OPTIMIZACIÓN: Chunk size reducido para respuestas más específicas
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)


# --- 3. FUNCIONES PRINCIPALES ---

def cargar_y_crear_indice(pdf_path: str):
    """Carga el PDF y crea un índice vectorial (si no existe) en ChromaDB."""
    print(f"📄 Cargando documento: {pdf_path}...")
    try:
        loader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = loader.load_data()
        
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo no encontrado. Revise la ruta: '{pdf_path}'")
        return None, None 
    except Exception as e:
        print(f"❌ ERROR: No se pudo cargar el archivo PDF.")
        print(f"Detalle del error: {e}")
        return None, None 

    print("🧠 Creando Base de Datos Vectorial (Chroma)...")

    # 1. Crear el cliente de ChromaDB y la colección
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection("pdf_index")

    # 2. Configurar el VectorStore de LlamaIndex
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 3. Crear el índice (esto genera los embeddings)
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
    )

    print("✅ Sistema indexado y listo.")
    return index, documents


def query_system(index, documents):
    """Ciclo interactivo para preguntas y resumen."""
    if index is None:
        return

    # Manejo de Preguntas Fuera de Contexto (OOC)
    OOC_ANSWER = "No se encontraron fragmentos de texto relevantes en el documento para responder a su pregunta."

    # OPTIMIZACIÓN: response_mode=COMPACT para velocidad en Q&A
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode=ResponseMode.COMPACT, 
        empty_response_answer=OOC_ANSWER
    )

    print("\n--- ¡BIENVENIDO! Sistema Q&A con LlamaIndex/Gemma 3:4b ---")
    print("   Escribe **'resumen'** para generar un resumen completo.")
    print("   Escribe **'salir'** para terminar.\n")

    while True:
        question = input("❓ Tu pregunta o comando: ")

        if question.lower() in ["salir", "exit"]:
            print("👋 Fin de la consulta.")
            break

        if question.lower() == "resumen":
            # Usamos TREE_SUMMARIZE para resúmenes largos y complejos
            print("\n⏳ Generando resumen completo del documento (Usando Gemma 3:4b)...")
            try:
                # Se crea un nuevo query engine para el modo de resumen
                summarizer = index.as_query_engine(response_mode=ResponseMode.TREE_SUMMARIZE)
                response = summarizer.query("Genera un resumen detallado y estructurado del documento proporcionado, resaltando los puntos clave.")

                print("\n📝 RESUMEN COMPLETO:")
                print(response.response)
                print("-" * 50)
            except Exception as e:
                print(f"❌ ERROR al generar resumen: {e}")
                print("Verifique la conexión con Ollama.")
            continue

        if not question.strip():
            continue

        print("... Buscando y generando respuesta (con Gemma 3:4b)...")

        try:
            response = query_engine.query(question)

            print("\n✅ RESPUESTA:")
            print(response.response)

            # Mostramos fuentes SÓLO si hay una respuesta de contexto
            if response.response != OOC_ANSWER:
                print("\n📚 FUENTE (Fragmentos del PDF usados para responder):")
                for node in response.source_nodes:
                    page_label = node.metadata.get('page_label', 'Pág. N/A')
                    source_file = node.metadata.get('file_name', 'Archivo N/A')
                    
                    print(f"  - Fuente: {source_file} ({page_label})")
                    print(f"    Fragmento: {node.text.strip()[:150]}...")

            print("-" * 50)
        except Exception as e:
            print(f"❌ ERROR al procesar la consulta: {e}")
            print("Verifique que Ollama esté activo.")


# --- 4. PUNTO DE ENTRADA ---
if __name__ == "__main__":
    index, documents = cargar_y_crear_indice(PDF_PATH)
    if index:
        query_system(index, documents)