import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

KB_PATH = "../knowledge_base"
DB_PATH = '../data/chroma/'

# Carga de documentos
def load_documents():
    loader = DirectoryLoader(KB_PATH, glob="*.txt")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap = 200
    )
    splits = splitter.split_documents(documents)

    return splits

docs = load_documents()

# Crea el vector
def create_db(docs):
    embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    
    vectorStore = Chroma.from_documents(
    documents = docs,
    embedding = embedding,
    persist_directory = DB_PATH
    )

    return vectorStore

vectorStore = create_db(docs)


def create_chain(vectorStore):
    model = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name = "gpt-3.5-turbo",
        temperature = 0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Responde las preguntas del usuario a partir del siguiente contexto: {context}. Sólo contesta por temas relacionados al contexto proporcionado. No intentes contestar sobre otros temas."),
            MessagesPlaceholder(variable_name = "chat_history"),
            ("human", "{input}")
        ]
    )

    # Cadena de documentos
    chain = create_stuff_documents_chain(
        llm = model,
        prompt = prompt
    )

    retriever = vectorStore.as_retriever()

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name = "chat_history"),
            ("human", "{input}"),
            ("system", "Dado el historial de la conversación, genera una nueva pregunta que busque información relevante para la conversación. No intentes contestar sobre temas que no están en la base de conocimiento.")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm = model,
        retriever = retriever,
        prompt = retriever_prompt
    )

    # Cadena de recuperación
    retrieval_chain = create_retrieval_chain(
        # retriever,
        history_aware_retriever,
        chain
    )

    return retrieval_chain

chain = create_chain(vectorStore)


def process_user_input(chain, question, chat_history):
    
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })


    return response["answer"]


chat_history = []


while True:
    user_input = input("Vos: ")

    if user_input.lower() == "salir":
        break

    response = process_user_input(chain, user_input, chat_history)
    chat_history.append(HumanMessage(content = user_input))
    chat_history.append(AIMessage(content = response))
    print("Asistente: ", response)


# question = "¿Cómo agregar un médico radiólogo?"