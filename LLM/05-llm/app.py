from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

# Ortam değişkenlerini yükle
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY bulunamadı.")


#Pdf dosyasini al documentsta isleme hazir hale getir
file_path = r"C:\Users\slayer\Desktop\KAİRU\LLM-portfolio\05-llm\ML 00 Makine Öğrenmesi Giriş.pdf"

if not os.path.exists(file_path) :
    raise FileNotFoundError(f"PDF  DOSYASI BULUNAMADI :  {file_path}")

loader = PyPDFLoader(file_path)
documents = loader.load()

#Metni kucuk parcalara ayiriyoruz cunku context windowa takilmak istemiyorum
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
text = splitter.split_documents(documents)

#daha sonra hizlica arama yapmak icin vektor veri tabanina saklama yapmak adina vector embedding yapiyoruz
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
db=FAISS.from_documents(text, embedding_model)

#llm
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    streaming=True,
    openai_api_key=openai_api_key
    )

#llm ile vektor veri tabanini birbirine baglayarak mekanizmayi olusturur
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm,  retriever=retriever)

#asistanınıza konuşma hafızası kazandırır
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

konular = {
    "makine oğrenmesi nedir?": "Makine Öğrenmesi Nedir?",
    "makine oğrenmesi uygulama adımları": "Makine Öğrenmesi Uygulama Adımları",
    "makine oğrenmesi türleri": "Makine Öğrenmesi Türleri",
    "makine oğrenmesinde performans paremetreleri": "Makine Öğrenmesinde Performans Paremetreleri"
}

# --------- Sohbet Başlat ---------
print("ML Mentor Asistanı Başladı (Çıkmak için 'çık' yazın)\n")

# Öğrenilen konular listesi
ogrenilen_konular = set()

while True:
    user_input = input("👤 Siz: ")
    if user_input.lower() in ["çık", "exit"]:
        print("Görüşmek üzere!")
        break

    # Geri Bildirim İsteği Kontrolü
    if "bugün ne öğrendim" in user_input.lower() or "geri bildirim" in user_input.lower():
        print("\n🤖 Asistan (Geri Bildirim):")

        if ogrenilen_konular:
            print("Bugün öğrendiğiniz konular:")
            for konu in ogrenilen_konular:
                print(f"- {konu}")
        else:
            print("- Henüz bir konu üzerinde konuşmadık.")

        eksik_konular = set(konular.values()) - ogrenilen_konular
        if eksik_konular:
            print("\nEksik kalan konular:")
            for konu in eksik_konular:
                print(f"- {konu}")
            print("\nBu konulara da göz atmanızı öneririm.")
        else:
            print("\nTebrikler! Tüm temel konuları öğrendiniz.")
        continue

    # QA Zincirinden cevap al
    response = qa_chain.invoke(user_input)
    print("🤖 Asistan:", response)

    # Konu çıkarımı
    for anahtar, konu in konular.items():
        if anahtar in user_input.lower():
            ogrenilen_konular.add(konu)