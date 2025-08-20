"""
Çok Dilli Hikaye Üretimi ve Sentiment Analiz Sistemi
Bu sistem, farklı dillerde hikaye üretimi, duygu analizi ve
metin kalitesi değerlendirmesi yapabilen gelişmiş bir NLP uygulamasıdır.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
    GenerationConfig,
    MarianMTModel,
    MarianTokenizer
)
import json
import time
import random
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade

class MultilingualStoryGenerator:
    """Çok dilli hikaye üretimi ve analiz sistemi"""
    
    def __init__(self, cache_dir: str = "./multilingual_cache"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model bileşenleri
        self.story_model = None
        self.story_tokenizer = None
        self.sentiment_analyzer = None
        self.translator = None
        
        # Veri saklama
        self.story_database = []
        self.analysis_results = []
        
        # Dil kodları
        self.supported_languages = {
            'tr': 'Türkçe',
            'en': 'English', 
            'de': 'Deutsch',
            'fr': 'Français',
            'es': 'Español'
        }
    
    def initialize_models(self):
        """Tüm modelleri yükler"""
        print("🤖 Modeller yükleniyor...")
        
        # Ana hikaye üretim modeli
        try:
            print("📚 Hikaye üretim modeli yükleniyor...")
            self.story_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.story_tokenizer.pad_token = self.story_tokenizer.eos_token
            
            self.story_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print("✅ Hikaye modeli yüklendi!")
            
        except Exception as e:
            print(f"❌ Hikaye modeli yüklenemedi: {e}")
            return False
        
        # Sentiment analiz modeli
        try:
            print("😊 Sentiment analiz modeli yükleniyor...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Sentiment analiz modeli yüklendi!")
            
        except Exception as e:
            print(f"⚠️ Sentiment analiz modeli yüklenemedi: {e}")
            # Basit alternatif
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                device=0 if torch.cuda.is_available() else -1
            )
        
        return True
    
    def create_story_prompts(self) -> Dict[str, List[str]]:
        """Farklı türlerde hikaye başlangıçları oluşturur"""
        return {
            "fantastik": [
                "Büyülü ormanın derinliklerinde yaşayan genç büyücü",
                "Ejderhaların son sığınağında beklenmedik bir keşif",
                "Kaybolmuş krallığın anahtarını bulan çoban kızı",
                "Zaman büyüsü ile geçmişe seyahat eden alchemist",
                "Ruhları görebilen gizemli yetenekli genç kadın"
            ],
            "bilim_kurgu": [
                "2157 yılında Mars kolonisinde yaşayan bilim insanı",
                "Yapay zeka ile dostluk kuran yalnız programcı",
                "Paralel evrenleri keşfeden kuantum fizikçisi",
                "Robotların hakimiyetindeki dünyada hayatta kalan",
                "Uzay gemisinde uyanıp hafızasını kaybetmiş kaptan"
            ],
            "macera": [
                "Kayıp hazinenin peşindeki cesur kaşif",
                "Tehlikeli dağlarda sıkışan dağcılar grubu",
                "Gizemli adada karaya vuran gemi mürettebatı",
                "Antik tapınağın labirentinde kaybolmuş arkeolog",
                "Sırrını korumak için kaçan eski casus"
            ],
            "romantik": [
                "Küçük kasabadaki kütüphanede tanışan iki yabancı",
                "Mektup arkadaşlığı ile başlayan aşk hikayesi",
                "Çocukluk arkadaşının yıllar sonra dönüşü",
                "Farklı dünyalardan gelen iki kalbin buluşması",
                "Geçmişin hatalarını telafi etmeye çalışan eski sevgili"
            ]
        }
    
    def generate_story(self, prompt: str, 
                      genre: str = "genel",
                      length_target: str = "medium",
                      creativity_level: float = 0.8) -> Dict[str, Any]:
        """Hikaye üretir ve detaylı analiz yapar"""
        
        # Uzunluk hedefleri
        length_settings = {
            "short": {"max_tokens": 100, "sentences": "3-5"},
            "medium": {"max_tokens": 200, "sentences": "6-10"}, 
            "long": {"max_tokens": 350, "sentences": "11-20"}
        }
        
        settings = length_settings.get(length_target, length_settings["medium"])
        
        print(f"📝 Hikaye üretiliyor: {genre} türü, {length_target} uzunluk")
        
        start_time = time.time()
        
        # Story prompt hazırlama
        formatted_prompt = f"Bu bir {genre} hikayesidir. {prompt}"
        
        # Tokenize
        inputs = self.story_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=settings["max_tokens"],
            temperature=creativity_level,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=self.story_tokenizer.pad_token_id,
            eos_token_id=self.story_tokenizer.eos_token_id
        )
        
        # Üretim
        with torch.no_grad():
            outputs = self.story_model.generate(
                **inputs,
                generation_config=gen_config,
                return_dict_in_generate=True
            )
        
        # Decode
        full_text = self.story_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        story_text = full_text[len(formatted_prompt):].strip()
        
        generation_time = time.time() - start_time
        
        # Hikaye analizi
        story_analysis = self._analyze_story_quality(story_text)
        
        # Sentiment analizi
        sentiment_result = self._analyze_sentiment(story_text)
        
        result = {
            "id": len(self.story_database) + 1,
            "prompt": prompt,
            "genre": genre,
            "length_target": length_target,
            "creativity_level": creativity_level,
            "story_text": story_text,
            "full_text": full_text,
            "generation_time": generation_time,
            "timestamp": datetime.now().isoformat(),
            "analysis": story_analysis,
            "sentiment": sentiment_result,
            "settings": settings
        }
        
        # Veritabanına ekle
        self.story_database.append(result)
        
        return result
    
    def _analyze_story_quality(self, text: str) -> Dict[str, Any]:
        """Hikaye kalitesini analiz eder"""
        
        # Temel metrikler
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Kelime istatistikleri
        word_count = len(words)
        unique_words = len(set(words))
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Cümle istatistikleri
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Okunabilirlik skorları
        try:
            readability_score = flesch_reading_ease(text)
            grade_level = flesch_kincaid_grade(text)
        except:
            readability_score = 0
            grade_level = 0
        
        # Karakter analizi
        char_mentions = len(re.findall(r'\b[A-Z][a-z]+\b', text))  # Büyük harfle başlayan kelimeler
        dialogue_count = text.count('"')  # Diyalog sayısı
        
        # Yaratıcılık göstergeleri
        adjectives = len(re.findall(r'\b\w+ly\b|\b\w+ing\b|\b\w+ed\b', text))
        complex_sentences = len(re.findall(r'[,;:]', text))
        
        return {
            "word_count": word_count,
            "unique_words": unique_words,
            "vocabulary_richness": unique_words / word_count if word_count > 0 else 0,
            "avg_word_length": avg_word_length,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "paragraph_count": len(paragraphs),
            "readability_score": readability_score,
            "grade_level": grade_level,
            "character_mentions": char_mentions,
            "dialogue_count": dialogue_count // 2,  # Çift olarak sayıldığı için
            "descriptive_elements": adjectives,
            "complex_sentences": complex_sentences,
            "story_quality_score": self._calculate_quality_score(
                word_count, unique_words, sentence_count, complex_sentences
            )
        }
    
    def _calculate_quality_score(self, word_count: int, unique_words: int, 
                               sentence_count: int, complex_sentences: int) -> float:
        """Hikaye kalite skoru hesaplar (0-100)"""
        
        # Çeşitli faktörlerin ağırlıklı ortalaması
        vocabulary_score = (unique_words / word_count * 100) if word_count > 0 else 0
        length_score = min(word_count / 150 * 100, 100)  # Optimal 150 kelime
        complexity_score = min(complex_sentences / sentence_count * 100, 100) if sentence_count > 0 else 0
        
        total_score = (vocabulary_score * 0.4 + length_score * 0.3 + complexity_score * 0.3)
        return min(total_score, 100)
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Metin duygu analizi yapar"""
        
        try:
            # Metni cümlelere böl
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
            
            sentence_sentiments = []
            
            for sentence in sentences[:10]:  # İlk 10 cümle
                try:
                    result = self.sentiment_analyzer(sentence)[0]
                    sentence_sentiments.append({
                        "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                        "label": result['label'],
                        "score": result['score']
                    })
                except:
                    continue
            
            # Genel duygu
            if sentence_sentiments:
                overall_sentiment = self.sentiment_analyzer(text[:512])[0]  # İlk 512 karakter
                
                # Duygu dağılımı
                sentiment_distribution = {}
                for sent in sentence_sentiments:
                    label = sent['label']
                    sentiment_distribution[label] = sentiment_distribution.get(label, 0) + 1
                
                return {
                    "overall_sentiment": overall_sentiment,
                    "sentence_sentiments": sentence_sentiments,
                    "sentiment_distribution": sentiment_distribution,
                    "emotional_complexity": len(set(s['label'] for s in sentence_sentiments))
                }
            else:
                return {"error": "Sentiment analizi yapılamadı"}
                
        except Exception as e:
            return {"error": f"Sentiment analiz hatası: {str(e)}"}
    
    def batch_story_generation(self, prompts_dict: Dict[str, List[str]], 
                             experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Toplu hikaye üretimi yapar"""
        
        print(f"📚 Toplu hikaye üretimi başlıyor...")
        total_stories = sum(len(prompts) for prompts in prompts_dict.values()) * len(experiments)
        print(f"🎯 Toplam üretilecek hikaye: {total_stories}")
        
        all_results = []
        story_count = 0
        
        for experiment in experiments:
            print(f"\n🧪 Deney parametreleri: {experiment}")
            
            for genre, prompts in prompts_dict.items():
                for prompt in prompts:
                    try:
                        result = self.generate_story(
                            prompt=prompt,
                            genre=genre,
                            **experiment
                        )
                        all_results.append(result)
                        story_count += 1
                        
                        print(f"✅ Hikaye {story_count}/{total_stories} tamamlandı")
                        
                        # İlerleme raporu
                        if story_count % 5 == 0:
                            avg_quality = np.mean([r['analysis']['story_quality_score'] for r in all_results])
                            print(f"📊 Ortalama kalite skoru: {avg_quality:.1f}")
                        
                    except Exception as e:
                        print(f"❌ Hikaye üretim hatası: {e}")
                        continue
        
        print(f"🎉 Toplu üretim tamamlandı! {len(all_results)} hikaye üretildi.")
        return all_results
    
    def analyze_story_collection(self, stories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Hikaye koleksiyonunu analiz eder"""
        
        print("📊 Hikaye koleksiyonu analiz ediliyor...")
        
        # Tür bazlı analiz
        genre_stats = defaultdict(list)
        for story in stories:
            genre_stats[story['genre']].append(story)
        
        # Genel istatistikler
        quality_scores = [s['analysis']['story_quality_score'] for s in stories]
        word_counts = [s['analysis']['word_count'] for s in stories]
        generation_times = [s['generation_time'] for s in stories]
        
        # En iyi hikayeler
        best_stories = sorted(stories, key=lambda x: x['analysis']['story_quality_score'], reverse=True)[:5]
        
        # Sentiment dağılımı
        sentiment_labels = []
        for story in stories:
            if 'overall_sentiment' in story['sentiment']:
                sentiment_labels.append(story['sentiment']['overall_sentiment']['label'])
        
        sentiment_dist = {}
        for label in sentiment_labels:
            sentiment_dist[label] = sentiment_dist.get(label, 0) + 1
        
        analysis = {
            "collection_summary": {
                "total_stories": len(stories),
                "genres": list(genre_stats.keys()),
                "avg_quality_score": np.mean(quality_scores),
                "avg_word_count": np.mean(word_counts),
                "avg_generation_time": np.mean(generation_times),
                "total_generation_time": sum(generation_times)
            },
            "genre_analysis": {
                genre: {
                    "count": len(stories_list),
                    "avg_quality": np.mean([s['analysis']['story_quality_score'] for s in stories_list]),
                    "avg_length": np.mean([s['analysis']['word_count'] for s in stories_list]),
                    "best_story_id": max(stories_list, key=lambda x: x['analysis']['story_quality_score'])['id']
                }
                for genre, stories_list in genre_stats.items()
            },
            "quality_distribution": {
                "excellent": len([s for s in stories if s['analysis']['story_quality_score'] >= 80]),
                "good": len([s for s in stories if 60 <= s['analysis']['story_quality_score'] < 80]),
                "average": len([s for s in stories if 40 <= s['analysis']['story_quality_score'] < 60]),
                "poor": len([s for s in stories if s['analysis']['story_quality_score'] < 40])
            },
            "sentiment_analysis": {
                "distribution": sentiment_dist,
                "dominant_sentiment": max(sentiment_dist.items(), key=lambda x: x[1])[0] if sentiment_dist else "unknown"
            },
            "best_stories": [
                {
                    "id": story['id'],
                    "genre": story['genre'],
                    "quality_score": story['analysis']['story_quality_score'],
                    "word_count": story['analysis']['word_count'],
                    "prompt": story['prompt'][:100] + "..."
                }
                for story in best_stories
            ]
        }
        
        return analysis
    
    def save_story_collection(self, stories: List[Dict[str, Any]], 
                            analysis: Dict[str, Any], 
                            filename_prefix: str = "story_collection") -> str:
        """Hikaye koleksiyonunu kaydeder"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Hikayeleri kaydet
        stories_file = f"{filename_prefix}_{timestamp}.json"
        with open(stories_file, 'w', encoding='utf-8') as f:
            json.dump(stories, f, ensure_ascii=False, indent=2)
        
        # Analizi kaydet
        analysis_file = f"analysis_{filename_prefix}_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        # Özet rapor oluştur
        report_file = f"report_{filename_prefix}_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🏆 HİKAYE KOLEKSİYONU RAPORU\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"📚 Toplam Hikaye: {analysis['collection_summary']['total_stories']}\n")
            f.write(f"🎭 Türler: {', '.join(analysis['collection_summary']['genres'])}\n")
            f.write(f"⭐ Ortalama Kalite: {analysis['collection_summary']['avg_quality_score']:.1f}\n")
            f.write(f"📝 Ortalama Uzunluk: {analysis['collection_summary']['avg_word_count']:.0f} kelime\n")
            f.write(f"⏱️  Toplam Süre: {analysis['collection_summary']['total_generation_time']:.1f} saniye\n\n")
            
            f.write("🏅 EN İYİ HİKAYELER:\n")
            f.write("-" * 30 + "\n")
            for i, story in enumerate(analysis['best_stories'], 1):
                f.write(f"{i}. {story['genre'].upper()} - Skor: {story['quality_score']:.1f}\n")
                f.write(f"   Prompt: {story['prompt']}\n\n")
        
        print(f"💾 Koleksiyon kaydedildi:")
        print(f"   📚 Hikayeler: {stories_file}")
        print(f"   📊 Analiz: {analysis_file}")
        print(f"   📋 Rapor: {report_file}")
        
        return timestamp
    
    def interactive_story_studio(self):
        """İnteraktif hikaye üretim stüdyosu"""
        
        print("\n🎨 İNTERAKTİF HİKAYE STÜDYOSU")
        print("=" * 50)
        print("Komutlar:")
        print("  /genres - Mevcut türleri göster")
        print("  /random - Rastgele hikaye üret")
        print("  /settings - Ayarları değiştir")
        print("  /library - Hikaye kütüphanesi")
        print("  /analyze - Son hikayeleri analiz et")
        print("  /quit - Çıkış")
        print("-" * 50)
        
        # Varsayılan ayarlar
        current_settings = {
            "genre": "fantastik",
            "length_target": "medium",
            "creativity_level": 0.8
        }
        
        genres = list(self.create_story_prompts().keys())
        session_stories = []
        
        while True:
            user_input = input(f"\n🎭 [{current_settings['genre']}] Hikaye prompt'u veya komut: ").strip()
            
            if user_input == "/quit":
                print("👋 Hikaye stüdyosu kapatılıyor...")
                if session_stories:
                    save_session = input("Bu oturumun hikayelerini kaydetmek ister misiniz? (y/n): ")
                    if save_session.lower() in ['y', 'yes', 'evet']:
                        session_analysis = self.analyze_story_collection(session_stories)
                        self.save_story_collection(session_stories, session_analysis, "session")
                break
                
            elif user_input == "/genres":
                print(f"\n🎭 Mevcut türler:")
                for i, genre in enumerate(genres, 1):
                    print(f"  {i}. {genre}")
                continue
                
            elif user_input == "/random":
                random_genre = random.choice(genres)
                prompts = self.create_story_prompts()[random_genre]
                random_prompt = random.choice(prompts)
                
                print(f"🎲 Rastgele: {random_genre} - {random_prompt}")
                
                story = self.generate_story(
                    prompt=random_prompt,
                    genre=random_genre,
                    **current_settings
                )
                self._display_story_result(story)
                session_stories.append(story)
                continue
                
            elif user_input == "/settings":
                print(f"\n⚙️ Mevcut ayarlar:")
                for key, value in current_settings.items():
                    print(f"  {key}: {value}")
                
                # Ayar değiştirme
                new_genre = input(f"Tür ({current_settings['genre']}): ").strip()
                if new_genre and new_genre in genres:
                    current_settings['genre'] = new_genre
                
                new_length = input(f"Uzunluk [short/medium/long] ({current_settings['length_target']}): ").strip()
                if new_length in ['short', 'medium', 'long']:
                    current_settings['length_target'] = new_length
                
                try:
                    new_creativity = input(f"Yaratıcılık [0.1-1.5] ({current_settings['creativity_level']}): ").strip()
                    if new_creativity:
                        creativity = float(new_creativity)
                        if 0.1 <= creativity <= 1.5:
                            current_settings['creativity_level'] = creativity
                except ValueError:
                    pass
                
                continue
                
            elif user_input == "/library":
                if session_stories:
                    print(f"\n📚 Bu oturumda {len(session_stories)} hikaye üretildi:")
                    for story in session_stories[-5:]:  # Son 5 hikaye
                        print(f"  🎭 {story['genre']} - Skor: {story['analysis']['story_quality_score']:.1f}")
                        print(f"     {story['prompt'][:80]}...")
                else:
                    print("📚 Henüz hikaye üretilmedi!")
                continue
                
            elif user_input == "/analyze":
                if len(session_stories) >= 3:
                    analysis = self.analyze_story_collection(session_stories)
                    print(f"\n📊 OTURUM ANALİZİ:")
                    print(f"📚 Toplam hikaye: {analysis['collection_summary']['total_stories']}")
                    print(f"⭐ Ortalama kalite: {analysis['collection_summary']['avg_quality_score']:.1f}")
                    print(f"📝 Ortalama uzunluk: {analysis['collection_summary']['avg_word_count']:.0f} kelime")
                    
                    if analysis['best_stories']:
                        best = analysis['best_stories'][0]
                        print(f"🏆 En iyi hikaye: {best['genre']} (Skor: {best['quality_score']:.1f})")
                else:
                    print("📊 Analiz için en az 3 hikaye gerekli!")
                continue
                
            elif user_input == "":
                continue
                
            else:
                # Hikaye üret
                try:
                    story = self.generate_story(
                        prompt=user_input,
                        **current_settings
                    )
                    self._display_story_result(story)
                    session_stories.append(story)
                    
                except Exception as e:
                    print(f"❌ Hikaye üretim hatası: {e}")
    
    def _display_story_result(self, story: Dict[str, Any]):
        """Hikaye sonucunu güzel formatta gösterir"""
        
        print(f"\n✨ HİKAYE #{story['id']} - {story['genre'].upper()}")
        print("=" * 60)
        print(f"📝 {story['story_text']}")
        print("\n" + "-" * 40)
        
        analysis = story['analysis']
        print(f"📊 Kalite Skoru: {analysis['story_quality_score']:.1f}/100")
        print(f"📏 Uzunluk: {analysis['word_count']} kelime, {analysis['sentence_count']} cümle")
        print(f"🎨 Kelime Çeşitliliği: {analysis['vocabulary_richness']:.2f}")
        print(f"⏱️  Üretim Süresi: {story['generation_time']:.2f} saniye")
        
        # Sentiment bilgisi
        if 'overall_sentiment' in story['sentiment']:
            sentiment = story['sentiment']['overall_sentiment']
            print(f"😊 Genel Duygu: {sentiment['label']} ({sentiment['score']:.2f})")

def main():
    """Ana program - Çok dilli hikaye üretim sistemi"""
    
    print("📚 ÇOK DİLLİ HİKAYE ÜRETİMİ VE ANALİZ SİSTEMİ")
    print("=" * 70)
    
    # Sistem başlatma
    generator = MultilingualStoryGenerator()
    
    # Modelleri yükle
    if not generator.initialize_models():
        print("❌ Modeller yüklenemedi, program sonlandırılıyor.")
        return
    
    print("\n🎯 Ne yapmak istiyorsunuz?")
    print("1. 🧪 Otomatik hikaye deneyimi çalıştır")
    print("2. 🎨 İnteraktif hikaye stüdyosu")
    print("3. 📊 Her ikisini de yap")
    
    choice = input("\nSeçiminiz (1-3): ").strip()
    
    if choice in ["1", "3"]:
        print("\n🧪 Otomatik hikaye deneyimi başlıyor...")
        
        # Prompt'ları hazırla
        story_prompts = generator.create_story_prompts()
        
        # Farklı deney parametreleri
        experiments = [
            {"length_target": "short", "creativity_level": 0.6},
            {"length_target": "medium", "creativity_level": 0.8},
            {"length_target": "long", "creativity_level": 1.0}
        ]
        
        # Deneyimi çalıştır
        stories = generator.batch_story_generation(story_prompts, experiments)
        
        # Analiz yap
        analysis = generator.analyze_story_collection(stories)
        
        # Sonuçları göster
        print(f"\n🏆 DENEY SONUÇLARI")
        print("=" * 50)
        summary = analysis['collection_summary']
        print(f"📚 Toplam Hikaye: {summary['total_stories']}")
        print(f"🎭 Türler: {', '.join(summary['genres'])}")
        print(f"⭐ Ortalama Kalite: {summary['avg_quality_score']:.1f}/100")
        print(f"📝 Ortalama Uzunluk: {summary['avg_word_count']:.0f} kelime")
        print(f"⏱️  Toplam Süre: {summary['total_generation_time']:.1f} saniye")
        
        # Tür bazlı performans
        print(f"\n🎭 TÜR BAZLI PERFORMANS:")
        print("-" * 40)
        for genre, stats in analysis['genre_analysis'].items():
            print(f"{genre.upper()}: {stats['avg_quality']:.1f} kalite, {stats['avg_length']:.0f} kelime")
        
        # En iyi hikayeler
        print(f"\n🏅 EN İYİ 3 HİKAYE:")
        print("-" * 30)
        for i, story in enumerate(analysis['best_stories'][:3], 1):
            print(f"{i}. {story['genre']} - {story['quality_score']:.1f} puan")
            print(f"   {story['prompt']}")
        
        # Duygu dağılımı
        if analysis['sentiment_analysis']['distribution']:
            print(f"\n😊 DUYGU DAĞILIMI:")
            print("-" * 25)
            for sentiment, count in analysis['sentiment_analysis']['distribution'].items():
                percentage = (count / summary['total_stories']) * 100
                print(f"{sentiment}: {count} hikaye ({percentage:.1f}%)")
        
        # Verileri kaydet
        timestamp = generator.save_story_collection(stories, analysis, "experiment")
        print(f"\n💾 Tüm veriler kaydedildi (ID: {timestamp})")
    
    if choice in ["2", "3"]:
        print(f"\n{'='*20} İNTERAKTİF STÜDYO {'='*20}")
        generator.interactive_story_studio()
    
    print(f"\n🎉 Program tamamlandı! Güzel hikayeler ürettiniz!")

if __name__ == "__main__":
    main()