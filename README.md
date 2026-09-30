# 🛍️ E-Commerce Semantic Search Chatbot

An AI-based customer support chatbot that uses **Natural Language Processing (NLP)** and **Semantic Search** to understand customer queries and automatically identify the correct category.

The chatbot is designed for e-commerce companies that receive thousands of customer queries every day.

---

## 📌 Problem Statement

E-commerce companies receive many customer messages related to:

- Order tracking
- Product returns
- Payment failures
- Refund status

Manually handling and categorizing these queries is time-consuming.

This project provides an automated chatbot that understands the customer's message, finds the most similar query from a document corpus, identifies the category, and provides an appropriate response.

---

## 🎯 Objectives

- Automatically classify customer queries.
- Use NLP for processing customer messages.
- Perform semantic search over a document corpus.
- Identify the most relevant category.
- Provide an automated customer-support response.
- Deploy the chatbot as a web application.

---

## 🧠 Technologies Used

- Python
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity
- HTML
- CSS
- JavaScript
- FastAPI
- Vercel

---

## 📂 Customer Query Categories

The chatbot currently supports four major categories:

| Category | Example |
|---|---|
| 🚚 Order Tracking | "Where is my order?" |
| 📦 Product Return | "I want to return my product." |
| 💳 Payment Failure | "My payment failed." |
| 💰 Refund Status | "When will I receive my refund?" |

---

## 🔍 How It Works

```text
Customer Query
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Find Most Similar Document
      ↓
Identify Category
      ↓
Generate Response
      ↓
Display Response
