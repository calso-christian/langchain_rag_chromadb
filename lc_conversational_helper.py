import os

from dotenv import load_dotenv
load_dotenv()


from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma

from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever

current_dir = os.path.dirname(os.path.abspath(__file__))
db_paths = {
    "metadata": os.path.join(current_dir, 'db','chroma_db_with_metadata'),
    "scraper": os.path.join(current_dir, 'db','chroma_db_scraper')
}


embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

retrievers = {
    name: Chroma(persist_directory=path,embedding_function=embeddings).as_retriever(search_type='similarity', search_kwargs={'k': 3})
    for name, path in db_paths.items()
}

multi_retrievers = EnsembleRetriever(retrievers=list(retrievers.values()))


# retriever = db.as_retriever(
#     search_type='similarity',
#     search_kwargs={'k':3,}
# )

model=ChatOpenAI(model='gpt-4o')

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    model, multi_retrievers, contextualize_q_prompt
)


qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If the question is not related to any of the retrieved context"
    "Say that you don't have the knowledge of the question"
    "If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


question_answer_chain = create_stuff_documents_chain(model,qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = [] 
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = rag_chain.invoke({"input": query, "chat_history": chat_history})

        print(f"AI: {result['answer']}")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

if __name__ == "__main__":
    continual_chat()