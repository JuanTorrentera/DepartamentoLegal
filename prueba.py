import os
import sys
import traceback
import hashlib
import uuid
import chromadb 

PdfReader = None
try:
    from llama_index.core.readers.file import PdfReader
    PdfReader = PdfReader
except Exception:
    try:
        from llama_index.readers.file import PdfReader
        PdfReader = PdfReader
    except Exception:
        PdfReader = None

DocumentClass = None
for mod_name in ("llama_index.schema", "llama_index.data_structs", "llama_index.core.schema"):
    try:
        mod = __import__(mod_name, fromlist=["Document"])
        if hasattr(mod, "Document"):
            DocumentClass = mod.Document
            break
    except Exception:
        pass

if DocumentClass is None:
    class DocumentClass:
        def __init__(self, text, metadata=None, doc_id=None):
            self.text = text or ""
            self.metadata = metadata or {}
            self.id_ = doc_id or str(uuid.uuid4())
            self.id = self.id_ 
            self.hash = hashlib.sha1(self.text.encode("utf-8")).hexdigest()

        def get_text(self):
            return self.text
        
        def get_metadata_str(self, mode=None):
            meta_parts = []
            if 'source' in self.metadata:
                meta_parts.append(f"Source: {self.metadata['source']}")
            if 'page' in self.metadata:
                meta_parts.append(f"Page: {self.metadata['page']}")
            return " | ".join(meta_parts)
            
        def __str__(self):
            return self.text

try:
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
except Exception:
    from llama_index import VectorStoreIndex, Settings

try:
    from llama_index.llms.ollama import Ollama
except Exception:
    Ollama = None

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except Exception:
    HuggingFaceEmbedding = None

try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
except Exception:
    ChromaVectorStore = None
PDF_PATH = r"C:\Users\juanm\OneDrive\Escritorio\Departamento legal\LIElec_090321.pdf"
CHROMA_DB_PATH = os.path.abspath("./chroma_db_llama")
LLM_MODEL_NAME = "gemma3:4b" 
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

GLOBAL_LLM_OBJECT = None
local_embed_model = None 

def get_chroma_client(persist_path: str):
    if chromadb is None:
        raise RuntimeError("chromadb no está instalado. Instálalo: pip install chromadb")
    return chromadb.PersistentClient(path=persist_path)

def extract_text_with_pypdf(pdf_path: str):
    """Extrae texto usando pypdf como fallback."""
    try:
        from pypdf import PdfReader as _PdfReader
    except Exception:
        raise RuntimeError("No está instalado pypdf. Instálalo: pip install pypdf")
    
    reader = _PdfReader(pdf_path)
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            text = p.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages

def build_documents_from_pdf(pdf_path: str):
    """Crea objetos Document a partir del PDF, usando el loader de LlamaIndex o pypdf."""
    if PdfReader is not None:
        try:
            loader = PdfReader()
            docs = loader.load_data(file=pdf_path)
            if docs:
                return docs
        except Exception:
            pass

    pages = extract_text_with_pypdf(pdf_path)
    docs = []
    for page_num, text in pages:
        metadata = {"page": page_num, "source": os.path.basename(pdf_path)}
        docs.append(DocumentClass(text=text, metadata=metadata))
    return docs

if Ollama is not None:
    try:
        globals()['GLOBAL_LLM_OBJECT'] = Ollama(model=LLM_MODEL_NAME, request_timeout=360.0)
        Settings.llm = globals()['GLOBAL_LLM_OBJECT'] 
    except Exception:
        pass

if HuggingFaceEmbedding is not None:
    try:
        globals()['local_embed_model'] = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        Settings.embed_model = globals()['local_embed_model']
    except Exception:
        pass 
def cargar_y_crear_indice(pdf_path: str):
    print(f" Cargando documento: {pdf_path} ...")
    if not os.path.isfile(pdf_path):
        print(f" ERROR: no existe el archivo PDF en la ruta: {pdf_path}")
        return None, None

    try:
        documents = build_documents_from_pdf(pdf_path)
    except Exception as e:
        print(" ERROR al extraer texto del PDF:", e)
        traceback.print_exc()
        return None, None

    print(" Preparando índice (intento persistente con Chroma)...")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    client = None
    collection = None
    if chromadb is not None:
        try:
            client = get_chroma_client(CHROMA_DB_PATH)
            collection = client.get_or_create_collection("pdf_index")
        except Exception:
            pass

    embed_model_instance = globals().get('local_embed_model')
    if embed_model_instance is None:
        print(" ERROR CRÍTICO: El modelo de Embeddings (HuggingFace) no pudo inicializarse.")
        return None, None

    if collection is not None and ChromaVectorStore is not None:
        try:
            vector_store = ChromaVectorStore(chroma_collection=collection)
            index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, embed_model=embed_model_instance)
            print(" Índice persistente (Chroma) creado y listo.")
            return index, documents
        except Exception:
            pass 

    print(" Creando índice EN MEMORIA (sin persistencia).")
    try:
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model_instance)
        print(" Índice en memoria creado y listo.")
        return index, documents
    except Exception as e:
        print(" ERROR al crear índice en memoria:", e)
        traceback.print_exc()
        return None, None

def query_system(index, documents):
    if index is None:
        return
        
    llm = globals().get('GLOBAL_LLM_OBJECT')
    if llm is None:
        print(" ERROR: El modelo Ollama no se inicializó correctamente.")
        return

    try:
        query_engine = index.as_query_engine(similarity_top_k=3, llm=llm) 
    except Exception as e:
        print(" ERROR al crear query engine:", e)
        traceback.print_exc()
        return

    print("\n--- Sistema Q&A con LlamaIndex/Gemma ---")
    print(" Escribe 'resumen' para un resumen completo, 'salir' para terminar.\n")

    while True:
        q = input(" pregunta: ").strip()
        if not q:
            continue
        if q.lower() in ("salir", "exit"):
            break
        
        if q.lower() == "resumen":
            try:
                summarizer = index.as_query_engine(response_mode="tree_summarize", llm=llm)
                resp = summarizer.query("Genera un resumen detallado del documento proporcionado.")
                text = getattr(resp, "response", None) or str(resp)
                print("\n RESUMEN:")
                print(text)
            except Exception as e:
                print(" ERROR al generar resumen:", e)
                traceback.print_exc()
            continue

        try:
            resp = query_engine.query(q)
            text = getattr(resp, "response", None) or str(resp)
            print("\n RESPUESTA:")
            print(text)
            
            print("\n FUENTES:")
            for node in getattr(resp, "source_nodes", []) or []:
                meta = getattr(node, "metadata", {}) or {}
                page = meta.get("page", meta.get("page_label", "N/A"))
                source = meta.get("file_name", meta.get("source", "N/A"))
                snippet = getattr(node, "text", "")[:200]
                print(f" - {source} (pág {page}): {snippet}...")
        except Exception as e:
            print(" ERROR al procesar la consulta:", e)
            traceback.print_exc()

if __name__ == "__main__":
    index, documents = cargar_y_crear_indice(PDF_PATH)
    
    if index and GLOBAL_LLM_OBJECT:
        query_system(index, documents)