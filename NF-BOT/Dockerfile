FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

# Atualize e instale as dependências
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip python3-venv

# Crie e ative um ambiente virtual
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Atualize o pip e instale os pacotes Python
RUN pip install --upgrade pip && \
    pip install openai langchain PyPDF2 faiss-cpu tiktoken langchain_openai langchain-community unidecode python-dotenv


# Limpe o cache de apt
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copie os arquivos para o container
COPY main.py /app/main.py
COPY arquivos /app/arquivos

WORKDIR /app

CMD ["python3", "main.py"]