import json
import os
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 발급받으신 OpenAI API 키를 입력해 주세요.
os.environ["GOOGLE_API_KEY"] = "AIzaSyB6nHqS7ZwhzTv_4oInCZRaY7lAbKFNDOA"
# 1. 정성껏 만든 펫테일 게코 JSON 데이터 불러오기
file_path = "gecko_morphs.json"

try:
    with open(file_path, "r", encoding="utf-8") as file:
        morph_data = json.load(file)
        
    # 1. JSON 데이터를 LangChain Document 객체로 변환
    docs = []
    for item in morph_data:
        # 검색 정확도를 높이기 위해 문장을 구조화하여 하나의 텍스트로 묶어줍니다.
        page_content = f"모프 이름: {item['morph_name_kr']} ({item['morph_name_en']})\n유전 형질: {item['genetics']}\n특징: {item['description']}\n주의사항: {item['caution']}"
        
        # 나중에 이미지 경로 등을 매칭하기 위해 metadata에 이름을 저장해 둡니다.
        doc = Document(page_content=page_content, metadata={"name": item['morph_name_en']})
        docs.append(doc)
        
    print("문서 변환 완료! 벡터 데이터베이스 생성을 시작합니다...")

    # 2. 임베딩 모델 및 벡터 데이터베이스 생성 (ChromaDB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    # 3. 검색기(Retriever) 설정 (가장 유사한 내용 2개를 찾아오도록 설정)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 4. LLM (언어 모델) 설정
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # 5. 프롬프트 템플릿 작성
    template = """
    당신은 펫테일 게코 전문가입니다. 아래 제공된 문맥(Context)을 바탕으로 질문에 정확하게 답변하세요.
    제공된 문맥에 없는 내용은 지어내지 말고 모른다고 답변하세요.

    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 6. RAG 체인 조립
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG 시스템 구축 완료! 질문을 테스트합니다.\n")
    print("-" * 50)

    # 7. 테스트 질문 던지기
    test_question = "화이트아웃 모프를 브리딩할 때 주의할 점이 뭐야?"
    print(f"질문: {test_question}")
    
    response = rag_chain.invoke(test_question)
    print(f"AI 답변: {response}")

except Exception as e:
    print(f"오류가 발생했습니다: {e}")
    
except FileNotFoundError:
    print("gecko_morphs.json 파일을 찾을 수 없습니다. 경로를 다시 확인해 주세요.")