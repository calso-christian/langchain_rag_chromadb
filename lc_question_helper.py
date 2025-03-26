import os
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()

from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class QueryInput(TypedDict):
    query: str

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db','chroma_db_with_metadata')

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


retriever = db.as_retriever(
    search_type='similarity',
    search_kwargs={'k':3,}
)

def retrieve_relevant_docs(query):
    relevant_docs=retriever.invoke(query)
    return relevant_docs

def format_input(query, relevant_docs):
    doc_contents='\n\n'.join([doc.page_content for doc in relevant_docs])


    return f"""
        Here are some documents that might help answer the question: {query}

        Relevant Documents: 
        {doc_contents}

        Please provide an answer based only on the provided documents.
        If the answer is not found, respond with 'I'm not sure'
    """
def create_chain():

    model = ChatOpenAI(model='gpt-4o-mini')

    def process_query(inputs: QueryInput):
        query = inputs["query"] 
        relevant_docs=retrieve_relevant_docs(query)
        combined_input=format_input(query,relevant_docs)

        messages = (
            SystemMessage(content='You are a helpful assistant'),
            HumanMessage(content=combined_input)
        )

        result = model.invoke(messages)
        return result.content
    
    return RunnableLambda(process_query).with_types(input_type=QueryInput)