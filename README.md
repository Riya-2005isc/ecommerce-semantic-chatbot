# 🛍️ E-Commerce Semantic Search Chatbot

An AI-based customer support chatbot that uses **Natural Language Processing (NLP)** and **Semantic Search** to understand customer queries and automatically identify the correct category.

The chatbot is designed for an e-commerce company that receives thousands of customer messages every day related to orders, returns, payments, and refunds.

---

## 📌 Project Overview

E-commerce companies receive a large number of customer queries every day.

Some common customer messages are:

- "Where is my order?"
- "I want to return my product."
- "My payment was unsuccessful."
- "When will I receive my refund?"

Manually reading and categorizing these messages can be time-consuming.

This project develops an **AI-based semantic search chatbot** that takes a customer message, compares it with a document corpus, identifies the most relevant category, and provides an appropriate response.

---

## 🎯 Problem Statement

To develop an AI-based chatbot that can understand customer messages, identify the relevant e-commerce support category using semantic similarity, and automatically provide a suitable response.

---

## 🎯 Objectives

The main objectives of this project are:

- To automate customer query classification.
- To apply Natural Language Processing to customer messages.
- To create a document corpus containing customer queries.
- To convert text into numerical representations using TF-IDF.
- To calculate similarity using Cosine Similarity.
- To identify the most relevant customer-support category.
- To provide an automated response.
- To develop a user-friendly chatbot interface.
- To deploy the chatbot as a web application.

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend and NLP processing |
| NLP | Processing customer messages |
| TF-IDF | Text vectorization |
| Cosine Similarity | Measuring text similarity |
| FastAPI | Backend API |
| HTML | Chatbot structure |
| CSS | User interface design |
| JavaScript | Chatbot interaction |
| Vercel | Deployment |
| GitHub | Source code management |

---

## 📂 Customer Query Categories

The chatbot currently supports four main categories:

| Category | Example Query |
|---|---|
| 🚚 Order Tracking | "Where is my order?" |
| 📦 Product Return | "I want to return my product." |
| 💳 Payment Failure | "My payment failed." |
| 💰 Refund Status | "When will I receive my refund?" |

---

# 🔬 Methodology

The chatbot follows an NLP-based semantic search workflow.

```text
              Customer Query
                    ↓
             Text Processing
                    ↓
             TF-IDF Vectorization
                    ↓
               Query Vector
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
```

### Step 1: Customer Query

The customer enters a question into the chatbot.

Example:

> "The payment failed but money was deducted."

### Step 2: Text Processing

The customer message is prepared for NLP processing.

The text is converted to lowercase and common stop words are handled by the TF-IDF vectorizer.

### Step 3: TF-IDF Vectorization

TF-IDF converts the customer query and the stored documents into numerical vectors.

TF-IDF stands for:

**Term Frequency - Inverse Document Frequency**

It helps identify the importance of words within the document corpus.

### Step 4: Cosine Similarity

The chatbot compares the customer query vector with the vectors of the stored documents.

Cosine Similarity is used to determine how similar the query is to each document.

### Step 5: Category Identification

The document with the highest similarity score is selected.

The category associated with that document becomes the predicted category.

### Step 6: Response Generation

The chatbot retrieves the predefined response associated with the predicted category and displays it to the customer.

---

# 📊 Document Corpus

A custom document corpus was created for this case study.

The corpus contains different customer queries belonging to four categories.

### 🚚 Order Tracking

Examples:

- Where is my order?
- How can I track my order?
- Where is my package?
- When will my order arrive?
- What is the status of my delivery?

### 📦 Product Return

Examples:

- I want to return my product.
- How can I return an item?
- I want to send back the product I purchased.
- Can I return my order?
- I need to request a product return.

### 💳 Payment Failure

Examples:

- My payment failed.
- My payment was unsuccessful.
- The payment did not go through.
- Why did my payment fail?
- Money was deducted but my order was not placed.

### 💰 Refund Status

Examples:

- Where is my refund?
- When will I receive my refund?
- My refund has not arrived.
- My refund is still pending.
- I returned my product but have not received the money.

---

# ⚙️ Text Processing

The chatbot processes customer messages before performing semantic search.

The main steps are:

1. Convert text to lowercase.
2. Handle common stop words.
3. Convert text into TF-IDF vectors.
4. Compare vectors using cosine similarity.
5. Select the most similar document.

---

# 🔢 TF-IDF Vectorization

TF-IDF is used to represent text numerically.

The implementation uses:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)
```

The `(1, 2)` setting allows the system to consider:

- Individual words
- Two-word combinations

For example:

```text
payment
payment failed
order
order tracking
product return
```

---

# 📐 Cosine Similarity

Cosine Similarity is used to measure the similarity between the customer query and documents in the corpus.

The formula is:

```text
              A · B
Similarity = -------
             ||A|| ||B||
```

Where:

- **A** = Customer query vector
- **B** = Document vector

A higher similarity score indicates that the query is more similar to the selected document.

> Note: The similarity score is not the same as model accuracy. It represents the similarity between the query and the selected document.

---

# 🔄 System Workflow

```text
User enters message
        ↓
Chatbot receives message
        ↓
Text preprocessing
        ↓
TF-IDF transformation
        ↓
Cosine similarity calculation
        ↓
Highest similarity selected
        ↓
Category identified
        ↓
Predefined response selected
        ↓
Response displayed to user
```

---

# 💬 Example

## Example 1: Payment Failure

### Customer Query

> The payment failed but the amount was deducted from my account.

### Chatbot Output

```text
Category: Payment Failure
```

### Response

> If the amount was deducted but the payment failed, please check your bank or payment app for the transaction status. If the amount is not automatically refunded, contact customer support with your transaction details.

---

## Example 2: Product Return

### Customer Query

> I want to send back the product I purchased.

### Chatbot Output

```text
Category: Product Return
```

### Response

> You can request a return from the Orders section and follow the return instructions.

---

## Example 3: Order Tracking

### Customer Query

> Where is my package?

### Chatbot Output

```text
Category: Order Tracking
```

### Response

> You can track your order using the tracking information provided in your order confirmation.

---

## Example 4: Refund Status

### Customer Query

> I returned my product but have not received the money.

### Chatbot Output

```text
Category: Refund Status
```

### Response

> Refunds are generally processed after the returned product is received and verified.

---

# 🧪 Testing

The chatbot was tested with different customer queries.

| Test Query | Expected Category |
|---|---|
| Where is my package? | Order Tracking |
| How can I track my order? | Order Tracking |
| I want to return my product. | Product Return |
| I want to send back my item. | Product Return |
| My payment failed. | Payment Failure |
| The payment did not go through. | Payment Failure |
| Where is my refund? | Refund Status |
| My refund has not arrived. | Refund Status |

The chatbot successfully identifies the relevant category for the tested queries.

---

# 🖥️ Chatbot Interface

The project includes a professional web-based chatbot interface.

The interface contains:

- Company-style branding
- Customer support chatbot
- Message input box
- Send button
- Category information
- Similarity score
- Automated chatbot responses
- Responsive design for different screen sizes

---

# 📸 Application Preview

Add your chatbot screenshot here.

For example:

```markdown
![Chatbot Interface](screenshots/chatbot.png)
```

You can create a folder named:

```text
screenshots
```

and upload your chatbot screenshot inside it.

---

# 📁 Project Structure

```text
ecommerce-semantic-chatbot/
│
├── api/
│   └── index.py
│
├── requirements.txt
│
├── vercel.json
│
├── README.md
│
└── screenshots/
    └── chatbot.png
```

### File Description

**`api/index.py`**

Contains the FastAPI backend, NLP processing, document corpus, TF-IDF vectorization, cosine similarity and chatbot interface.

**`requirements.txt`**

Contains the Python dependencies required for the project.

**`vercel.json`**

Contains the Vercel deployment configuration.

**`README.md`**

Contains the project documentation.

**`screenshots/`**

Contains screenshots of the chatbot interface.

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Riya-2005isc/ecommerce-semantic-chatbot.git
```

Move into the project directory:

```bash
cd ecommerce-semantic-chatbot
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running Locally

Run the FastAPI application using:

```bash
uvicorn api.index:app --reload
```

The application can then be opened in the browser using the local URL provided by FastAPI.

---

# 🚀 Deployment

The chatbot is deployed using **Vercel**.

### Deployment Workflow

```text
GitHub Repository
        ↓
Connect GitHub Repository to Vercel
        ↓
Vercel Build
        ↓
Deploy Application
        ↓
Live Chatbot
```

The project source code is maintained on GitHub and the application is deployed through Vercel.

---

# 🌐 Live Demo

### Vercel Deployment

👉 https://ecommerce-semantic-chatbot-three.vercel.app/

You can open the above link to interact with the chatbot.

---

# 📊 Results

The chatbot was tested using different customer messages.

The system successfully:

- Accepts natural-language customer queries.
- Converts text into numerical vectors.
- Calculates similarity with the document corpus.
- Identifies the relevant category.
- Generates an appropriate response.
- Displays the similarity score.

The chatbot demonstrates how semantic search can be used to automate basic e-commerce customer support.

---

# ⚠️ Limitations

The current version has some limitations:

- It supports a limited number of categories.
- Responses are predefined.
- It does not connect to a real order database.
- It does not provide real-time order information.
- TF-IDF may not fully understand complex language or context.

---

# 🔮 Future Scope

The chatbot can be improved by adding:

- A larger customer-support corpus.
- More e-commerce categories.
- Sentence embeddings.
- FAISS-based semantic search.
- Multilingual support.
- Real-time order tracking.
- Database integration.
- Conversation history.
- Voice-based customer support.
- Advanced transformer-based NLP models.

---

# 🔐 API Key

This project does **not require an OpenAI API key or external AI API key**.

The semantic search is performed locally using:

```text
TF-IDF + Cosine Similarity
```

---

# 📚 Key Concepts

The main concepts demonstrated in this project are:

- Natural Language Processing
- Text Preprocessing
- Document Corpus
- TF-IDF
- Vectorization
- Cosine Similarity
- Semantic Search
- Text Classification
- FastAPI
- Web Deployment

---

# 👩‍💻 Author

**Riya Rathod**

E-Commerce Semantic Search Chatbot

---

# ⭐ Acknowledgement

This project was developed as an NLP case study to demonstrate the practical application of semantic search for automated e-commerce customer support.

---

## 📄 License

This project is created for educational and academic purposes.
