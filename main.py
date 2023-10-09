import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ['OPENAI_API_KEY'] = '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'

st.title("InfoScope: Your News Insight Hub 📈")
st.sidebar.title("Article Catalog")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

process_url_clicked = st.sidebar.button('Check!')
file_path = 'news_research.pkl'
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:

    #loading data
    loader = UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Fetching the Latest Insights... 🚀")
    data = loader.load()

    #Splitting data
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',','], chunk_size = 1000)
    main_placeholder.text("Text Parsing in Progress... 🔠🔡")
    docs = text_splitter.split_documents(data)

    #Creating Embeddings and Saving to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Creating Semantic Signatures... 📊📈")
    time.sleep(2)

    #Saving the FAISS index as a local Pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input('Ask Me Anything')

if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vectorstore.as_retriever())
            result = chain({'question':query}, return_only_outputs=True)
            st.header('Answer')
            st.subheader(result['answer'])

            #Display Sources, if available

            sources = result.get('sources', '')
            if sources:
                st.subheader('Sources')
                sources_list = sources.split('\n')
                for source in sources_list:
                    st.write(source)

                    









