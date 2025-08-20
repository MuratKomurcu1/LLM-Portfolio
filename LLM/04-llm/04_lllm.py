import os
from dotenv import load_dotenv
import pandas as pd
from typing import List
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage

# --- 1. Yapılandırma ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OpenAI API anahtarı .env dosyasında bulunamadı.")

# --- 2. Sabitler ---
EXCEL_DOSYA_YOLU = r"C:\Users\fb\Desktop\KAİRU\LLM-portfolio\04-llm\04-LLM.xlsx"
SORU_SUTUNU_ADI = "Soru"
CEVAP_SUTUNU_ADI = "Cevap"

# --- 3. Veri Yükleme ve İşleme ---
def load_and_chunk_data(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
    df = pd.read_excel(file_path, engine='openpyxl').dropna(subset=[SORU_SUTUNU_ADI, CEVAP_SUTUNU_ADI])
    print(f"'{os.path.basename(file_path)}' yüklendi, {len(df)} geçerli satır bulundu.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    all_chunks = []
    for _, row in df.iterrows():
        combined_text = f"Soru: {row[SORU_SUTUNU_ADI]}\nCevap: {row[CEVAP_SUTUNU_ADI]}"
        chunks = splitter.split_text(combined_text)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append(Document(page_content=chunk_text, metadata={'original_question': row[SORU_SUTUNU_ADI]}))
    print(f"Toplam {len(all_chunks)} adet chunk oluşturuldu.")
    return all_chunks

# --- 4. Pipeline Kurulumu ---
all_chunks = load_and_chunk_data(EXCEL_DOSYA_YOLU)
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
print("Vektör veritabanı oluşturuluyor...")
db = FAISS.from_documents(all_chunks, embedding_model)
print("Vektör veritabanı (FAISS) başarıyla oluşturuldu!")
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=openai_api_key)

# --- 5. Alaka Filtresi (LLM as Judge) Zinciri ---
# Bu zincir, bulunan dokümanların kullanıcı sorusu için uygun olup olmadığına karar verir.
relevance_filter_prompt_template = """
Kullanıcının Sorusu: {question}

Bulunan Doküman:
---
{context}
---

GÖREV: Yukarıdaki 'Bulunan Doküman'ın, 'Kullanıcının Sorusu'nu DOĞRUDAN cevaplamak için alakalı olup olmadığını değerlendir.
Cevabın SADECE 'Evet' veya 'Hayır' olmalıdır.

Alakalı mı? (Evet/Hayır):
"""
relevance_prompt = PromptTemplate(
    template=relevance_filter_prompt_template,
    input_variables=["question", "context"]
)
relevance_filter_chain = LLMChain(llm=llm, prompt=relevance_prompt)

# --- 6. Akıllı Sorgu ve Cevap Döngüsü (YENİ MANTIK) ---
print("\n📘 Öncelikli Yerel Bilgi Asistanı Başlatıldı. Çıkmak için 'çık' yazın.\n")
while True:
    user_query = input("❓ Soru: ")
    if user_query.lower().strip() in ["çık", "exit", "quit"]:
        print("🛑 Asistan kapatıldı.")
        break

    try:
        # Adım 1: Yerel veritabanında en benzer dokümanı bul.
        candidate_docs = db.similarity_search(user_query, k=1)
        
        if not candidate_docs:
            print("\n⚠️ Yerel veritabanında ilgili hiçbir doküman bulunamadı. LLM'e soruluyor...")
            # Doğrudan LLM'e git, çünkü aday bile yok.
            messages = [
                SystemMessage(content="Sen yardımsever bir asistansın."),
                HumanMessage(content=user_query)
            ]
            llm_result = llm.invoke(messages)
            print("\n💬 Cevap (LLM Tarafından Üretildi):\n", llm_result.content)
            print("-" * 50)
            continue

        best_doc = candidate_docs[0]

        # Adım 2: LLM Yargıç'a alaka düzeyini sor.
        relevance_result = relevance_filter_chain.invoke({
            "question": user_query,
            "context": best_doc.page_content
        })
        is_relevant = "evet" in relevance_result['text'].lower()
        
        print(f"--- Yargıç Kararı: Bulunan içerik alakalı mı? -> {relevance_result['text'].strip()} ---")

        # Adım 3: Yargıç kararına göre hareket et.
        if is_relevant:
            # ALAKALIYSA: LLM KULLANMA. Cevabı doğrudan veritabanından sun.
            print("✅ Karar: İçerik alakalı. Cevap doğrudan yerel veritabanından sunuluyor...")
            print("\n💬 Cevap:\n", best_doc.page_content)
            print("-" * 50)
            print(f"Kaynak: Yerel Veritabanı | Orijinal Soru: {best_doc.metadata.get('original_question', 'N/A')}")
            print("-" * 50)
        else:
            # ALAKALI DEĞİLSE: LLM KULLAN.
            print("\n❌ Karar: İçerik alakasız. Cevap LLM ile üretiliyor...")
            messages = [
                SystemMessage(content="Sen yardımsever bir asistansın."),
                HumanMessage(content=user_query)
            ]
            llm_result = llm.invoke(messages)
            print("\n💬 Cevap (LLM Tarafından Üretildi):\n", llm_result.content)
            print("-" * 50)
            print("Not: Bu bilgi yerel veritabanında bulunamadığı için LLM tarafından genel bilgiyle cevaplanmıştır.")
            print("-" * 50)

    except Exception as e:
        print(f"⚠️ Hata oluştu: {e}")
