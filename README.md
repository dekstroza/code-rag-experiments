# RAG Creation with existing code base


## Requirements

1. Poetry (https://python-poetry.org/docs/) - Dependency Management
2. Python 3.8 or higher 
3. Ollama (https://ollama.com/) - For running LLM models
4. Huggingface cli (https://huggingface.co/docs/huggingface_hub/en/guides/cli)

## Installation

Clone the repository, and inside the repository run:
```bash
mkdir -p vectorstores/db 
mkdir source_code 
```

This is where the vector store will be created, and where you should clone
source repositories. Clone any code you wish to work with into the source_code
folder.

Install the Ollama for your operating system. You can find the installation
instructions at https://ollama.com/docs/installation.html. Once Ollama is
running, you can download the models you wish to use locally with:

```bash
ollama run llama3 
```
Above will make sure the model is downloaded and ready to be used. You dont need
to run it with ollama run llama3 when you are using the chatbot.

Embedding models are huggingface models, so you can download them using the cli
tool:
```bash

huggingface-cli login #(in case you need to log in, some models require
                      #verification through the login, in order to download them)
huggingface-cli download BAAI/bge-large-en-v1.5 #will download the model locally
``` 
See documentation on huggingface and ollama for more details on how to use it.

## Usage

Go to the root folder of this repository and run:
```bash
poetry install
```
This will install all the dependencies required for the project. Once completed,
run:
```bash
poetry shell
``` 
This will activate the virtual environment.

Once the virtual environment is activated, you can run the following command to:
```bash
./run model
```

This will open web browser (http://localhost:8000) where you can interact with the model.

## Changing the model:

Ollama can run many different models. To change the model, you can edit the file
chatbot.py. The model is kept in the variable `MODEL`. You can change this to
use different models.


## Short explanation:

Code is using chainlit to present UI. See chainlit website for more details on
how to use it. Interaction with the model through RAG is done with langchain
(see langchain documentation for more details).

For most basic loading of the code see the: java_code_loader.py  file. This file
will also attempt to load metadata json files, those are created with
parse_all_java_except_tests.py (see python code for details, its all simple
stuff).
For most basic maven site documentation loading see: html_loader.py
