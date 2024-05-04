import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import VectorDBQA


PATH = "knowledge_base"

def load_documents():
    loader = DirectoryLoader(PATH, glob="*.txt")
    documents = loader.load()
    return documents

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(load_documents())

embeddings = OpenAIEmbeddings(open_api_key=os.environ['OPENAI_API_KEY'])

doc_search = Chroma.from_documents(texts, embeddings)


qa = VectorDBQA.from_chain_type(llm = OpenAI(), chain_type = "stuff", vectorstore = doc_search, return_source_documents=True)
query = "¿Cómo puedo agregar 2 médicos radiólogos?"
result = qa({"query": query})


print(result['result'])
print(result['source_documents'])