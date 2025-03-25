import os
from dotenv import load_dotenv
load_dotenv()


from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db','chroma_db_with_metadata')

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


retriever = db.as_retriever(
    search_type='similarity',
    search_kwargs={'k':1,}
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
def generate_response(query):

    relevant_docs=retrieve_relevant_docs(query)
    combined_input=format_input(query,relevant_docs)

    model = ChatOpenAI(model='gpt-4o-mini')

    messages = (
        SystemMessage(content='You are a helpful assistant'),
        HumanMessage(content=combined_input)
    )

    result = model.invoke(messages)
    return result.content




if __name__ == "__main__":
    user_query = input('Enter your question: ')

    response=generate_response(user_query)

    print('\n--- Generated Response ---\n')
    print(response)