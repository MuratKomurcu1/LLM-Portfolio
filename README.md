# 🤖 LLM Portfolio - Machine Learning & AI Projects

## 📋 Proje Özeti

Bu repository, makine öğrenmesi ve büyük dil modelleri (LLM) alanındaki çalışmalarımı içeren kapsamlı bir portfolio projesidir. Projede temel ML kavramlarından ileri seviye NLP uygulamalarına kadar geniş bir yelpazede örnekler bulunmaktadır.

## 🎯 Ana Bileşenler

### 1. 📚 Temel Machine Learning Çalışmaları
- **01-LLM.py**: Transformers kütüphanesi ile temel tokenization ve sentiment analizi
- **Temel Tanımlar.docx**: LLM terminolojisi ve kavramsal açıklamalar
- **ML 00 Makine Öğrenmesi Giriş.pdf**: Kapsamlı ML teorik bilgiler

### 2. 🧠 LangChain ve Chain-of-Thought 
- **02-LLM.py**: LangChain ile CoT (Chain of Thought) implementasyonu
- OpenAI API entegrasyonu
- Problem çözme algoritmalarında sistematik yaklaşım

### 3. 🎨 Çok Dilli Hikaye Üretimi Sistemi
- **llm_inference_system.py**: Ana sistem dosyası
- Çoklu dil desteği (TR, EN, DE, FR, ES)
- Sentiment analizi ve hikaye kalitesi değerlendirmesi
- İnteraktif hikaye stüdyosu

### 4. 📊 Gradio ile Model Deployment
- **pathgradio.py**: Türkçe müşteri hizmetleri chatbot
- BERT tabanlı intent classification
- Web arayüzü ile canlı demo

### 5. 🔍 RAG (Retrieval Augmented Generation) Sistemi
- **04_lllm.py**: FAISS vectorstore ile belge arama
- **app.py**: PDF tabanlı soru-cevap sistemi
- Makine öğrenmesi dökümanlarından anlık bilgi çekme

### 6. 🌐 AI Code Assistant (Docker ile)
- FastAPI backend
- Streamlit frontend
- Kod analizi ve güvenlik taraması
- Docker compose ile kolay deployment

## 🛠️ Teknoloji Stack

### Core Libraries
```
- transformers: 4.35.2
- torch: 2.7.1
- langchain: 0.3.26
- openai: 1.97.0
- gradio: Latest
- streamlit: 1.28.0
- fastapi: 0.104.1
```

### ML/AI Components
- **Modeller**: GPT-4, BERT, RoBERTa
- **Vector Store**: FAISS
- **Embeddings**: OpenAI Embeddings
- **Languages**: Python, JavaScript

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
```bash
# Ana gereksinimler
pip install -r requirements.txt

# Çok dilli sistem için
pip install -r requirements_multilingual.txt
```

### 2. Ortam Değişkenleri
```bash
# .env dosyası oluşturun
OPENAI_API_KEY=your_api_key_here
```

### 3. Docker ile Çalıştırma
```bash
docker-compose up -d
```

### 4. Bireysel Projeler
```bash
# Hikaye üretimi sistemi
python llm_inference_system.py

# Gradio chatbot
python pathgradio.py

# RAG sistemi
python app.py
```

## 📁 Proje Yapısı

```
├── 📄 01-LLM.py                    # Temel örnekler
├── 📄 02-LLM.py                    # LangChain CoT
├── 📄 04_lllm.py                   # RAG sistemi
├── 📄 app.py                       # PDF QA sistemi
├── 📄 pathgradio.py               # Gradio chatbot
├── 📄 llm_inference_system.py     # Ana hikaye sistemi
├── 📊 04LLM.xlsx                  # Veri dosyası
├── 🐳 docker-compose.yml
├── 🐳 Dockerfile
├── 📋 requirements*.txt
└── 📄 README.md
```

## ✨ Özellikler

### 🎨 Hikaye Üretimi Sistemi
- **Çok dilli destek**: 5 farklı dilde hikaye üretimi
- **Tür çeşitliliği**: Fantastik, bilim kurgu, macera, romantik
- **Kalite analizi**: Otomatik metin kalitesi değerlendirmesi
- **Sentiment analiz**: Duygu durumu analizi
- **İnteraktif stüdyo**: Gerçek zamanlı hikaye üretimi

### 🤖 Chatbot Sistemi
- **Intent Classification**: Müşteri sorularını kategorize etme
- **Türkçe optimizasyonu**: BERT-base-turkish-cased model
- **Web arayüzü**: Gradio ile kullanıcı dostu interface

### 📚 RAG Sistemi
- **Belge işleme**: PDF'lerden bilgi çıkarma
- **Vector arama**: FAISS ile hızlı benzerlik araması
- **Soru-cevap**: Doğal dil ile sorgu yapabilme
- **Konversasyonel hafıza**: Geçmiş konuşmaları saklama


### Bilinen Sınırlamalar
- OpenAI API bağımlılığı
- Türkçe modellerde sınırlı performans
- GPU gereksinimi (isteğe bağlı)

## 📈 Öğrenim Hedefleri

Bu proje aşağıdaki konularda derinlemesine deneyim sağlar:
- **Transformer mimarisi** ve kullanımı
- **LangChain** framework'ü ile uygulama geliştirme
- **RAG sistemleri** tasarımı ve implementasyonu
- **Çok dilli NLP** uygulamaları
- **Model deployment** ve productionization
- **Docker** ile containerization


## 📄 Lisans

Bu proje MIT lisansı altında paylaşılmıştır.


**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**
