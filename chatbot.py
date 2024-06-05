from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from chromadb.config import Settings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import OllamaEmbeddings
from threading import Lock

import os
import tempfile

### Answer question ###
system_prompt = (
    "You are coding assistant and knowledge expert on java framework called ServiceFramework, or swfk for short.\n\n"
    "Assume ServiceFramework is built using Java 8, JEE and CDI, and it is not using Spring or Spring Boot related code at all. \n\n"
    "Assume applications built with ServiceFramework are always EAR or WAR. \n\n"
    "Assume build system is always Maven.\n\n"
    "Assume starting with ServiceFramework applications  is always done using ServiceFramework provided maven archetypes for EAR or WAR services. \n\n"
    "Assume Java version is always 8. \n\n"
    "Assume applications built with ServiceFramework are always deployed on JBoss JEE server. \n\n"
    "Answer the user's questions based on the below context:"
    "\n\n"
    "{context}"
)

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

MODEL = "llama3"

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

### Statefully manage chat history ###
store = {}

store_lock = Lock()

vectorstore = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    with store_lock:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
    return store[session_id]


def load_llm():
    llm = Ollama(model=MODEL, verbose=True,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),)
    return llm


def retrieval_qa_chain(llm,vectorstore):
    retriever=vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def qa_bot():
    llm=load_llm()
    DB_PATH = "vectorstores/db/"
    model_kwargs = {'device': 'mps','trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings( model_name="BAAI/bge-large-en-v1.5", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    #embedding = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding, client_settings=Settings(anonymized_telemetry=False, is_persistent=True,))
    qa = retrieval_qa_chain(llm,vectorstore)
    return qa

def process_file_uploads(message_elements):
    temp_dir = tempfile.gettempdir()
    for element in message_elements:
        temp_file = os.path.join(temp_dir, element.name)
        with open(temp_file, "wb") as file:
            file.write(element.content)
        vectorize_file(element.name)
        os.remove(temp_file)


def vectorize_file(file_name):
    # Load and process the documents
    loader = GenericLoader.from_filesystem(
        tmpfile.gettempdir(),
        glob=file_name,
        #suffixes=[".java"],
        parser=LanguageParser(language=Language.JAVA, parser_threshold=10),
    )
    docs = loader.load()

    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA,
        chunk_size=10000,
        chunk_overlap=200
    )
    result = java_splitter.split_documents(docs)
    vectorstore.add_documents(result)


@cl.on_chat_end
def on_chat_end():
    session_id = cl.user_session.get("id")
    print("The user disconnected!")
    print("Disconnected session id:", session_id)
    with store_lock:
        if session_id in store:
            del store[session_id]
    print("Session history deleted!")

@cl.on_chat_start
async def start():
          chain=qa_bot()
          msg=cl.Message(content="Starting up ServiceFramework Model...")
          await msg.send()
          msg.content= "Hi, welcome to ServiceFramework Bot. How can I help you?"
          await msg.update()
          cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
          answer_prefix_tokens=["FINAL", "ANSWER"]
          if message.elements:
             print("File was attached")
             java_files = message.elements
             with open(java_files[0].path, "r") as f:
                  java_file_content = f.read()
                  java_file_name = os.path.basename(java_files[0].path)
                  print("111:"+ java_files[0].name)
                  message.content = message.content + (" .Include content of this file in the context: "+ java_file_content)
        # Do something with java_file_content
          runnable = cl.user_session.get("chain")
          session_id = cl.user_session.get("id")
          cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=answer_prefix_tokens,)
          msg = cl.Message(content="")
          async for chunk in runnable.astream({"input": message.content},config=RunnableConfig(callbacks=[cb], configurable={"session_id": session_id},)):
                if "context" in chunk:
                    docs = chunk["context"]
                    docs_dict = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
                    chunk["context"] = docs_dict
                if "answer" in chunk:
                    answer = chunk["answer"]
                    await msg.stream_token(answer)
          await msg.send()
