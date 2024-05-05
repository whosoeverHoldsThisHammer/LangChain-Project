import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name = "gpt-3.5-turbo",
    temperature = 0.7
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Cuenta un chiste sobre un el siguiente tema"),
        ("human", "{input}")
    ]
)


chain = prompt | llm

response = chain.invoke({"input": "perro"})
print(response)