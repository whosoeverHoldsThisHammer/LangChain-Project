import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

doc = Document(
    page_content = "Para dar de alta a un médico..."

)

model = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name = "gpt-3.5-turbo",
    temperature = 0.1
)

prompt = ChatPromptTemplate.from_template("""
Responde la pregunta del usuario:
Context: {context}
Pregunta: {input}
""")

chain = prompt | model

response = chain.invoke({
    "input": "¿Cómo agregar un médico radiólogo?",
    "context": [doc]
})

print(response)