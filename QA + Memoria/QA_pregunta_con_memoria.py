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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

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


# Crea el retriever
retriever = vectordb.as_retriever(),


# Indicación para reformular las preguntas cuando hay un historial de conversación
instruction_to_system = """
A partir del historial de una conversación y de la última pregunta que hizo el usuario
que puede referenciar contexto en el historial, formula una nueva pregunta
que pueda ser entendida sin el historial de la conversación. NO contestes la pregunta
reformúla la pregunta si es necesario y si no, devuelve la pregunta tal como la hizo el usuario
"""


question_maker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_to_system),
        MessagesPlaceholder(variable_name = "chat_history"),
         ("human", "{question}"),
    ]
)


question_chain = question_maker_prompt | llm | StrOutputParser()

# history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

# question_chain = contextualize_q_prompt | llm | StrOutputParser()


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return question_chain
    else:
        return input["question"]


retriever_chain = RunnablePassthrough.assign(
    context = contextualized_question | retriever
)


# Indicación para el sistema de QA
qa_system_prompt = """
Eres un asistente que contesta preguntas sobre una base de conocimiento. \
Emplea los siguientes fragmentos de contexto para responder la pregunta. \
No intentes contestar sobre temas que no están en la base de conocimiento. \
Si no sabes qué contestar, pide disculpas.

{context}
Pregunta: {question}
Respuesta: 
"""


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


rag_chain = (
    retriever_chain
    | qa_prompt
    | llm
    | output_parser
)


chat_history = []

question = "¿Dónde puedo gestionar el alta de un paciente?"
ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content= question), ai_msg])

print(ai_msg.content)

 
question = "¿Podrías explicar con más detalle?"
ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content= question), ai_msg])

print(ai_msg.content)


# Crea las cadenas
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# chat_history = []

# question = "¿Dónde puedo gestionar el alta de un paciente?"

# Hace una primera pregunta (el historial está vacío)
# ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})

# chat_history.extend([HumanMessage(content = question), ai_msg_1["answer"]])

# print(ai_msg_1)