from langchain.document_loaders import UnstructuredMarkdownLoader, UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdfs = [

]
annual_reports = []
for pdf in pdfs:
    #loader = UnstructuredMarkdownLoader(pdf) MarkDownFromChatGpt
    loader = UnstructuredEPubLoader(pdf)
    # Load the PDF document
    document = loader.load()        
    # Add the loaded document to our list
    annual_reports.append(document)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

chunked_annual_reports = []
for annual_report in annual_reports:
    # Chunk the annual_report
    texts = text_splitter.split_documents(annual_report)
    # Add the chunks to chunked_annual_reports, which is a list of lists
    chunked_annual_reports.append(texts)
    print(f"chunked_annual_report length: {len(texts)}")
#Everything above should be split into ingesting phase
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.autonotebook import tqdm
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
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

# Upsert annual reports to Pinecone via LangChain.
# There's likely a better way to do this instead of Pinecone.from_texts()
for chunks in chunked_annual_reports:
    Pinecone.from_texts([chunk.page_content for chunk in chunks], embeddings, index_name=index_name) 

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
