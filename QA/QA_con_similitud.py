import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import VectorDBQA

KB_PATH = "knowledge_base"
DB_PATH = 'data/chroma/'
RELEVANCE_SCORE = 0.4

# Carga de documentos
def load_documents():
    loader = DirectoryLoader(KB_PATH, glob="*.txt")
    documents = loader.load()
    return documents

# Transformación de los documentos
def text_split():
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    return text_splitter.split_documents(load_documents())

# Vectorización
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

vectordb = Chroma.from_documents(
    documents=text_split(),
    embedding=embedding,
    persist_directory=DB_PATH
)

# Búsqueda en la base
def search_db(query):
    results = vectordb.similarity_search_with_relevance_scores(query, k=3)
    show_results(results)

# Control para ver si hay resultados y para ver si el puntaje de relevancia del primer resultado es mayor al umbral definido
def show_results(results):
    if len(results) == 0 or results[0][1] < RELEVANCE_SCORE:
        print("No encontré resultados relevantes")
    else:
        print(results)


query_1 = "¿Cómo puedo agregar 2 médicos radiólogos?"
query_2 = "¿Dónde puedo gestionar el alta de un paciente?"

search_db(query_2)