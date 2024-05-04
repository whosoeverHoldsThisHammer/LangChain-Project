import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain


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
    documents = text_split(),
    embedding = embedding,
    persist_directory = DB_PATH
)

# Inicialización del modelo
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)


# Define un template
template = """
Eres un experto en contestar las consultas que nuestros usuarios hacen acerca de nuestra base de conocimiento. Emplea los siguientes fragmentos de contexto para responder la pregunta.
1. Tu respuesta debería tener una longitud parecida a la de los fragmentos de contexto.
2. Presenta la respuesta en forma de lista.
3. Se respetuoso. No contestes a insultos.
4. No intentes contestar sobre temas que no están en la base de conocimiento.
5. Si no sabes qué contestar, pide disculpas.
6. Despidete diciendo hasta luego!
{context}
Pregunta: {question}
Respuesta: """

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Inicialización de una cadena con prompt
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = vectordb.as_retriever(),
    return_source_documents = True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Inicialización de la cadena sin prompt
# qa_chain = RetrievalQA.from_chain_type(
#    llm,
#    retriever= vectordb.as_retriever(),
#    return_source_documents = True
#)

# Búsqueda en la base
def search_db(query):
    results = qa_chain( {"query": query})
    show_results(results)


def show_results(results):
        print(results['result'])
        #print(results['source_documents'])


query_1 = "¿Cómo puedo agregar 2 médicos radiólogos?"
query_2 = "¿Dónde puedo gestionar el alta de un paciente?"

search_db(query_2)