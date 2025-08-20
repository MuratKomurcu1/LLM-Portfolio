

import csv
csv.field_size_limit(10000000)  # Çok uzun soru içerikleri için limit yükseltme

from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import gradio as gr

# 1. Veri setini yükle (ilk 2000 örnek, performans için sınırlı)
dataset = load_dataset("bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset", split="train[:2000]")
dataset = dataset.train_test_split(test_size=0.2)
label_names = sorted(list(set(dataset["train"]["intent"])))  # Intent kategorileri

# Label encoding - string'leri sayıya çevir
def encode_labels(example):
    example["labels"] = label_names.index(example["intent"])
    return example

# Her örneğe label numarası ekle
dataset = dataset.map(encode_labels)

# 2. Tokenizer ve model (Türkçe BERT)
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_names))

# 3. Tokenizasyon
def tokenize(batch):
    return tokenizer(batch["instruction"], padding="max_length", truncation=True)

encoded_dataset = dataset.map(tokenize, batched=True)




# 5. PyTorch tensör formatına çevir
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 6. Değerlendirme metrikleri
accuracy = load("accuracy")
f1 = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels),
        "f1": f1.compute(predictions=predictions, references=labels, average="macro"),
    }

# 7. Eğitim parametreleri
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # evaluation_strategy yerine eval_strategy
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 8. Trainer oluştur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics,
)

# 9. Modeli eğit
trainer.train()

# 10. Gradio arayüzü
def predict_label(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            intent = label_names[prediction]
            
            # Türkçe cevaplar
            turkish_responses = {
                "ORDER_STATUS": "📦 Siparişinizin durumunu kontrol ediyorum. Lütfen sipariş numaranızı paylaşır mısınız?",
                "SHIPPING_INFO": "🚚 Kargo bilgilerinizi gösteriyorum. Hangi konuda yardımcı olabilirim?",
                "RETURN_POLICY": "↩️ İade politikamız: 14 gün içinde ücretsiz iade hakkınız bulunmaktadır.",
                "PRODUCT_INFO": "🛍️ Ürün bilgileri hakkında size yardımcı olabilirim. Hangi ürünü merak ediyorsunuz?",
                "PAYMENT_INFO": "💳 Ödeme ile ilgili sorularınızı yanıtlayabilirim. Ne öğrenmek istiyorsunuz?",
                "ACCOUNT_ISSUES": "👤 Hesap sorunlarınız için size yardımcı olabilirim. Sorunuzu detaylandırır mısınız?",
                "DISCOUNT_INFO": "🎁 İndirim ve kampanyalarımız hakkında bilgi verebilirim.",
                "TECHNICAL_SUPPORT": "🔧 Teknik destek için buradayım. Hangi sorunu yaşıyorsunuz?",
                "COMPLAINT": "😔 Şikayetinizi dinliyorum. Lütfen yaşadığınız sorunu detaylı anlatın.",
                "COMPLIMENT": "😊 Güzel sözleriniz için teşekkür ederiz! Memnuniyetiniz bizim için çok değerli."
            }
            
            response = turkish_responses.get(intent, f"Bu soru '{intent}' kategorisine aittir. Size nasıl yardımcı olabilirim?")
            return f"🤖 Kategori: {intent}\n\n💬 Cevap: {response}"
            
    except Exception as e:
        return f"❌ Hata oluştu: {str(e)}"


gr.Interface(
    fn=predict_label,
    inputs=gr.Textbox(lines=3, label="Sorunuzu Türkçe yazın", placeholder="Örn: Siparişim ne zaman gelecek?"),
    outputs=gr.Textbox(label="Chatbot Cevabı", lines=4),
    title="🇹🇷 Türkçe Müşteri Hizmetleri Chatbot",
    description="Müşteri sorularınızı yazın, yapay zeka kategorisini tespit edip Türkçe cevap versin!"
).launch()