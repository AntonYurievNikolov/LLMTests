from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.autonotebook import tqdm
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import gradio as gr
import pinecone
import os


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
embeddings = OpenAIEmbeddings(
                                openai_api_key=OPENAI_API_KEY,
                                )
# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "esoteric"



vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)


# Create the chain
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
                temperature=0, 
                model_name="gpt-3.5-turbo"
                #top_p= 1,
                #frequency_penalty= 0,
                #presence_penalty= 0
                ), 
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)
# Initialize chat history list
chat_history = []

def ask_question(query):
    result = qa({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    chat_history.append((query, answer))
    return answer

iface = gr.Interface(fn=ask_question, inputs="text", outputs="text", title="Esoteric Q&A")
iface.launch()
# Add the answer to the chat history
