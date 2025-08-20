#TASK 1.
#----------------------------------------------------------------------------------------------------------
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Transformers are amazing!"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", token_ids)

# 4 tane token var. Model anlayabileceği şekilde 4 tane küçük metin parçası seçmiş
# Seçilen küçük metin parçalarına (tokenlara) karşılık gelen sayısal değerler tokenids olarak adlandırılıyor.

#------------------------------------------------------------------------------------------------------------

#TASK 2.
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I love using transformers!"))
#[{'label': 'POSITIVE', 'score': 0.9994327425956726}]
print(classifier("I will be a good coder in the future!"))
#[{'label': 'POSITIVE', 'score': 0.9998165965080261}]
print(classifier("I hate using ai!"))
#[{'label': 'NEGATIVE', 'score': 0.9970158338546753}]

#çok yakın bir şekilde doğru sonuçlar veriyor.

#------------------------------------------------------------------------------------------------------------
#TASK 3.

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("After understanding the logic very well about LLM", max_length=30, temperature=0.9)
print(result[0]['generated_text'])
print("------------------------------------------------------------------------------------------------------------")
result = generator("when ı wake up", max_length=30, temperature=0.3)
print(result[0]['generated_text'])
print("------------------------------------------------------------------------------------------------------------")
result = generator("I want coffee but I drank too much caffeine", max_length=30, temperature=0.7)
print(result[0]['generated_text'])
print("------------------------------------------------------------------------------------------------------------")

# Aynı ayarlarda 3 farklı başlangıç cümlesi ile denendi. Her seferinde farklı sonuçlar veriyor.Anlamlar bazen karışık olabiliyor.
#Önce hepsi 0.7 ile denendi. Sonra 0.3 ile denendi. 0.3 ile denendiğinde neredeyse tek kelime üretti.
#0.9 ile denendiğinde ise daha fazla kelime üretti. 0.9 ile denendiğinde daha fazla kelime ürettiği için daha açıklanır bir sonuç verdi.
#0.3 ile denendiğinde ise daha az kelime ürettiği için anlamı daha net fakat hikayesiz tekrarlamalar oldu.

#------------------------------------------------------------------------------------------------------------
#TASK 4.   
import os
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasındaki API anahtarını yükle
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt cümlesi
prompt = "Bir sabah uyandığında her şey değişmişti çünkü artık kimse konuşmuyor, herkes sadece bakışlarla iletişim kuruyordu."

# API üzerinden cevap oluştur
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  
    messages=[
        {"role": "system", "content": "Sen yaratıcı bir hikaye anlatıcısısın."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=256,
)

# Cevabı yazdır
print("🚀 Tamamlanan Metin:\n")
print(response.choices[0].message.content)

#Genç bir kadın olan Elif, sabah uyandığında alışık olmadığı bir sessizlikle karşılaştı. Sokakta yürürken fark etti ki
#insanlar birbirleriyle konuşmuyor, sadece göz teması kurarak iletişim kuruyorlardı. Bu durum Elif'i oldukça endişelendirdi
#ve etrafındaki insanları izlemeye başladı.Birkaç gün boyunca bu sessizlik devam etti. Elif, insanların gözlerinde farklı 
#duyguları ve mesajları görmeye başladı. Kimi insanlar endişeliydi, kimi mutluydu, kimi ise korkmuştu. Gözlerin ifade ettiği 
#duygular, insanların iç dünyalarını daha derinlemesine keşfetmesine olanak sağlıyordu.Elif, bu yeni iletişim şekline alışmaya
#başladıkça, insanların birbirlerine nasıl daha anlayışlı ve duyarlı olabileceğ

#Sonuç beni çok şaşırmadı çünkü OpenAI'nin modelleri genellikle yaratıcı ve anlamlı metinler üretebiliyor.Kelimeleri sınırladığım
#için daha kısa bir metin üretti. Eğer kelime sınırını artırmış olsaydım daha uzun ve detaylı bir hikaye yazabilirdi belki o zaman
#daha da şaşırabilirdim.