import os
import base64
import requests
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from unidecode import unidecode
from langchain_community.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import tiktoken

# Define o caminho para as pastas
pasta_arquivos = '/app/arquivos'
pasta_imagens = '/app/arquivos'

# Define a chave da API
os.environ["OPENAI_API_KEY"] = "sk-proj-oT1k1dJ4OJ6ZRJOkStpuT3BlbkFJtfp9gpnC5XWuLzb2IodX"

# Listar todos os arquivos na pasta
arquivos = os.listdir(pasta_arquivos)
arquivos_imagem = [arquivo for arquivo in arquivos if arquivo.lower().endswith(('.png', '.jpg', '.jpeg'))]
arquivos_pdf = [arquivo for arquivo in arquivos if arquivo.lower().endswith('.pdf')]

if arquivos_pdf:
    primeiro_pdf = arquivos_pdf[0]
    caminho_primeiro_pdf = os.path.join(pasta_arquivos, primeiro_pdf)

    # Processamento de PDF
    pdfreader = PdfReader(caminho_primeiro_pdf)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    def count_tokens(messages, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)
        total_tokens = 0
        for message in messages:
            total_tokens += len(encoding.encode(message))
        return total_tokens

    token_counts = count_tokens(texts)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type='stuff')

    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), db.as_retriever())
    chat_history = []

    queries = [
        "Número da fatura:",
        "Data de vencimento:",
    ]

    results = {}
    for query in queries:
        result = qa({"question": query, "chat_history": chat_history})
        results[query] = result['answer']

    json_dados = json.dumps(results, indent=4, ensure_ascii=False)
    with open('dados.json', 'w', encoding='utf-8') as file:
        file.write(json_dados)
    print("JSON salvo no arquivo 'dados.json'.")

elif arquivos_imagem:
    primeiro_imagem = arquivos_imagem[0]
    caminho_primeiro_imagem = os.path.join(pasta_imagens, primeiro_imagem)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(caminho_primeiro_imagem)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "Você é um assistente especializado em leitura de documentos. Responda exatamente apenas com o que foi pedido"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Retorne as seguintes informações, mantendo o formato original: Número da fatura:, Data de vencimento:."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    content = response_json['choices'][0]['message']['content']

    data = {}
    lines = content.split('\n')

    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            key = key.strip()
            value = value.strip()

            if key.startswith('- '):
                key = key[2:].strip()

            key = unidecode(key)
            value = unidecode(value)

            data[key] = value

    with open('dados.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print("Dados salvos em 'dados.json'.")

else:
    print('Não há arquivos PDF ou imagens na pasta especificada.')
