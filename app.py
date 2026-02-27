import streamlit as st
import json
import os
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 안전한 API 키 입력
current_dir = os.path.dirname(os.path.abspath(__file__))
password_file_path = os.path.join(current_dir, "api_key_security.json")
    
with open(password_file_path, "r", encoding="utf-8") as file:
  api_key = json.load(file)

real_api_key = api_key["api_key"]

os.environ["GOOGLE_API_KEY"] = real_api_key

# 2. 시스템 초기화 및 캐싱 (DB처럼 매번 로드하지 않도록 메모리에 유지합니다)
@st.cache_resource
def init_rag_system():
    
    file_path = os.path.join(current_dir, "gecko_morphs.json")
    
    with open(file_path, "r", encoding="utf-8") as file:
        morph_data = json.load(file)

    docs = []
    for item in morph_data:
        page_content = f"모프 이름: {item['morph_name_kr']} ({item['morph_name_en']})\n유전 형질: {item['genetics']}\n특징: {item['description']}\n주의사항: {item['caution']}"
        docs.append(Document(page_content=page_content, metadata={"name": item['morph_name_en']}))

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    template = """당신은 펫테일 게코 전문가입니다. 제공된 문맥(Context)을 바탕으로 질문에 정확하게 답변하세요.
    Context: {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    return ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# --- 웹 화면 UI 구성 시작 ---
st.title("🦎 펫테일 게코 AI 백과사전")
st.write("펫테일 게코의 모프 정보나 브리딩 주의사항을 자유롭게 물어보세요!")

# 백그라운드에서 RAG 시스템 준비
rag_chain = init_rag_system()

# 사용자 입력창 만들기
user_question = st.text_input("질문을 입력하세요 (예: 고스트 모프의 특징이 뭐야?)")

# 질문하기 버튼을 눌렀을 때의 동작
if st.button("AI에게 물어보기"):
    if user_question:
        with st.spinner("AI가 꽁꽁의 질문에 대한 답변을 열심히 찾고 있습니다..."):
            response = rag_chain.invoke(user_question)
            st.success("답변이 완료되었습니다!")
            st.write(response)
    else:
        st.warning("질문을 먼저 입력해 주세요!")