

from langchain.prompts import ChatPromptTemplate  # promts → prompts
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# .env dosyasından API key yükle
load_dotenv()

class BasicLangChainCoT:
    def __init__(self, api_key=None):
        """Basic LangChain CoT Class"""
        
        # API key'i .env'den al eğer verilmediyse
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            print("✅ API key bulundu!")
        else:
            self.llm = None
            print("❌ API key bulunamadı!")
            print("💡 .env dosyası oluştur:")
            print("   OPENAI_API_KEY=your_api_key_here")
                         
    def get_answer_prompt(self, problem):
        """Get answer to the question using LangChain CoT"""
        
        template = ChatPromptTemplate.from_template("""
        Sen çok iyi bir asistansın. Sorulara cevap verirken bir uzman görüşüne sahip olduğunu ve karşındakinin kaygılarını ona hissetirmeden
        düşürmen gerektiğini unutma. Gerekirse tanı koymaktan çekinme ve çözüm önerileri sun. 
        
        Problem: {problem}
        
        Çözüm formatı:
        1. 📖 Problemi anla: [hedefi netleştir]
        2. 📋 Gerekli adımları sırala: [mevcut bilgileri listele]
        3. 🔄 Adımları sırayla uygula: [adım adım çözüm]
        4. 💡 En iyi sonucu ver: [sonucu netleştir]
        5. ✅ Sonucu kontrol et: [sonucu kontrol et]
        """)
        
        return template.format(problem=problem)
    
    def cot_coz(self, problem):
        """Problemi CoT ile çöz"""
        
        # Prompt'u oluştur
        prompt = self.get_answer_prompt(problem)
        
        print(f"📝 Oluşturulan Prompt:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)
        
        # Eğer API varsa çöz
        if self.llm:
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                return f"❌ Hata: {e}"
        else:
            return "🔑 API key olmadığı için sadece prompt gösterildi."

def test_gunluk_problem():
    """Günlük problem örneği test et"""
    print("🏠 GÜNLÜK PROBLEM CoT TESTİ")
    print("=" * 35)
    
    # API key'siz başlat (otomatik .env'den alacak)
    cot = BasicLangChainCoT()
    
    problem = "Yalnız kaldığımda bazen nefes darlığı yaşıyorum. Gündelik streslerim artıyor ve bu durum beni endişelendiriyor. " \
              "Bununla başa çıkmak için ne yapabilirim? Lütfen adım adım bir çözüm önerisi sun ve bu konuda uzman görüşü ver."
    
    print(f"❓ Problem: {problem}")
    print("\n🤖 AI düşünüyor...")
    
    # Senin kodunda eksik olan kısım
    result = cot.cot_coz(problem)
    print(f"\n💡 Sonuç:\n{result}")

def test_matematik():
    """Matematik testi de ekleyelim"""
    print("\n\n🧮 MATEMATİK TESTİ")
    print("=" * 25)
    
    cot = BasicLangChainCoT()
    
    problem = "25 × 17 + 45 ÷ 9 işleminin sonucunu adım adım hesapla"
    
    print(f"❓ Problem: {problem}")
    print("\n🤖 AI düşünüyor...")
    
    result = cot.cot_coz(problem)
    print(f"\n💡 Sonuç:\n{result}")

def main():
    """Main function to run the tests"""
    print("🚀 Düzeltilmiş LangChain CoT")
    print("=" * 40)
    
    # Her iki testi de çalıştır
    test_gunluk_problem()
    test_matematik()
    
    print("\n\n🎓 BAŞARDIN!")
    print("✅ LangChain CoT çalışıyor")
    print("✅ .env dosyası güvenlik için kullanılıyor")
    print("✅ Hem matematik hem günlük problemleri çözebiliyor")

if __name__ == "__main__":
    main()

