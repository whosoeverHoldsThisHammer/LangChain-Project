import os
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = llm.invoke("Hola, cómo estás?")
print(response)