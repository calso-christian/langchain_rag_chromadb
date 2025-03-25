import os
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir,'books')
db_dir = os.path.join(current_dir,'db')
#file_path = os.path.join(current_dir, 'books','odyssey.txt')
persistent_directory = os.path.join(current_dir, 'db','chroma_db_with_metadata')


if not os.path.exists(persistent_directory):
    print('Persistent directory does not exist. Initializing Vector Store...')

    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            print(f"The file {books_dir} does not exist.")
        )
    
    book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]

    documents=[]

    for book_file in book_files:
        file_path=os.path.join(books_dir,book_file)
        loader=TextLoader(file_path,encoding="utf-8")
        book_docs=loader.load()
        for doc in book_docs:
            doc.metadata = {'source':book_file}
            documents.append(doc)
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    ) 
    print("\n--- Finished creating embeddings ---")
        
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")