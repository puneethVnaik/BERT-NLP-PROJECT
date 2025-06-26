# 🧠 Contextual Language Understanding with Transformer Models : Elevating NLP Capabilities

## 📌 Overview
This project focuses on building a **context-aware sentiment analysis system** using **BERT (Bidirectional Encoder Representations from Transformers)**. Designed as part of my internship at Rooman Technologies, it leverages transformer models to analyze and classify text with deeper contextual understanding, moving beyond traditional NLP methods.

The system includes a **Streamlit-based user interface** and a **FastAPI backend**, providing real-time sentiment classification and a seamless user experience.

---

## ✨ Features
✅ Context-Aware Sentiment Analysis using BERT  
✅ Real-time Predictions via Web UI (Streamlit)  
✅ FastAPI Backend for Efficient Inference  
✅ Supports Text Classification & Question Answering  
✅ Modular & Scalable Design for Future Expansion  

---

## 🎯 Motivation
Traditional NLP systems often fail to capture **context, sarcasm, and nuanced sentiment**. By implementing BERT, this project demonstrates how transformer-based models can improve prediction accuracy and deepen natural language understanding in real-world applications like reviews, chatbots, and opinion mining.

---

## 🚀 How It Works
1️⃣ **User Input** – Enter a sentence or paragraph via the web UI  
2️⃣ **API Call** – Text is sent to FastAPI backend as a POST request  
3️⃣ **Tokenization** – Input is processed using HuggingFace's BERT tokenizer  
4️⃣ **Model Inference** – Fine-tuned BERT model predicts sentiment  
5️⃣ **Output Display** – Results shown on UI with confidence score and visual cues  

---

## 🔧 Technologies Used

| Category             | Tools / Libraries                           |
|----------------------|---------------------------------------------|
| 💬 NLP Model         | BERT (HuggingFace Transformers)             |
| 🧠 Deep Learning     | PyTorch                                     |
| 🌐 API Framework     | FastAPI                                     |
| 🎛️ UI Framework      | Streamlit                                   |
| 📦 Data              | IMDB Movie Review Dataset                   |
| 🔧 Utilities         | NumPy, Pandas, Tokenizers                   |

---

## 🏗️ System Architecture
<pre><code> [User Input] ↓ [Streamlit UI] ↓ [FastAPI Backend] ↓ [BERT Tokenizer] ↓ [Fine-tuned BERT Model] ↓ [Prediction Output] ↓ [Streamlit Display UI] </code></pre>
