import os
from dotenv import load_dotenv

def run():
    print("Hola mundo")
    load_dotenv()
    print(os.environ['OPENAI_API_KEY'])


if __name__ == "__main__":
    run()