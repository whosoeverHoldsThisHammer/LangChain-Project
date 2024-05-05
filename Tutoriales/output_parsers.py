import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

model = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name = "gpt-3.5-turbo",
    temperature = 0.7
)


# 1 - String
def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Contame un chiste sobre el siguiente tema"),
            ("human", "{input}")
        ]
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain.invoke({
        "input": "Payasos"
    })

print(call_string_output_parser())


# 2 - Lista
def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Genera una lista de 5 sinónimos a partir de la siguiente palabra. Devolvé los resultados separados por comas"),
        ("human", "{input}")
    ]
)
        
    parser = CommaSeparatedListOutputParser()

    chain = prompt | model | parser

    return chain.invoke({
        "input": "Feliz"
    })

print(call_list_output_parser())


# 3 json
def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extraé la información de la siguiente frase. \n Instrucciones de formato: {format_instructions}"),
            ("human", "{frase}")
        ]
    )

    class Persona(BaseModel):
        nombre: str = Field(description = "Nombre de la persona") 
        edad: int = Field(description = "Edad de la persona")

    parser = JsonOutputParser(pydantic_object = Persona)

    chain = prompt | model | parser

    return chain.invoke({
        "frase": "Juan tiene 30 años",
        "format_instructions": parser.get_format_instructions()
    })

print(call_json_output_parser())


# 4 Otro json
def call_json_output_parser_alt():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extraé la información de la siguiente frase. \n Instrucciones de formato: {format_instructions}"),
            ("human", "{frase}")
        ]
    )

    class Pizza(BaseModel):
        receta: str = Field(description = "Nombre de la receta") 
        ingredientes: list = Field(description = "Ingredientes")

    parser = JsonOutputParser(pydantic_object = Pizza)

    chain = prompt | model | parser

    return chain.invoke({
        "frase": "Los ingredientes para una Pizza Margarita son: tomate, queso y albahaca",
        "format_instructions": parser.get_format_instructions()
    })

print(call_json_output_parser_alt())