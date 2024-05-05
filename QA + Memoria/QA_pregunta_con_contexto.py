import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage


KB_PATH = "../knowledge_base"
DB_PATH = '../data/chroma/'

# Carga de documentos
def load_documents():
    loader = DirectoryLoader(KB_PATH, glob="*.txt")
    documents = loader.load()
    return documents

# Transformación de los documentos
def text_split():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    return text_splitter.split_documents(load_documents())


# Vectorización
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# Guarda el vector en una base de datos
vectordb = Chroma.from_documents(
    documents = text_split(),
    embedding = embedding,
    persist_directory = DB_PATH
)

# Inicialización del modelo
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)

output_parser = StrOutputParser()

retriever = vectordb.as_retriever(),


instruction_to_system = """
A partir del historial de una conversación y de la última pregunta que hizo el usuario
que puede referenciar contexto en el historial, formula una nueva pregunta
que pueda ser entendida sin el historial de la conversación. NO contestes la pregunta
reformúla la pregunta si es necesario y si no, devuelve la pregunta tal como la hizo el usuario
"""

question_maker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_to_system),
        MessagesPlaceholder("chat_history"),
         ("human", "{question}"),
    ]
)

question_chain = question_maker_prompt | llm | StrOutputParser()

pregunta = question_chain.invoke({
    "question": "Podrías explicar con más detalle?",
    "chat_history": [HumanMessage(content="Explicaste cómo dar de alta a un paciente")]
})

print(pregunta)