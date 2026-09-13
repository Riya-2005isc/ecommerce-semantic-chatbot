import os
import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# OpenAI API client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --------------------------------------------------
# E-COMMERCE SUPPORT DOCUMENT CORPUS
# --------------------------------------------------

documents = [
    "Customers can track their order using the tracking ID provided after shipment.",
    "Orders that have been shipped can be monitored through the delivery tracking page.",

    "Customers can request a return within 7 days if the product meets the return conditions.",
    "To return an eligible item, open the order details and select the return option.",
    "Customers who want to send back a purchased product should select the return option in their order details.",

    "If a payment fails, check your payment method and try again.",
    "If money is deducted but the order is not placed, the transaction may be reversed automatically.",

    "Refunds are processed after the returned product is received and verified.",
    "The refund amount may take several business days to appear in the customer's bank account."
]

categories = [
    "Order Tracking",
    "Order Tracking",

    "Product Return",
    "Product Return",
    "Product Return",

    "Payment Failure",
    "Payment Failure",

    "Refund Status",
    "Refund Status"
]

# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return np.array(
        [item.embedding for item in response.data],
        dtype="float32"
    )


# Create document embeddings once
document_embeddings = get_embeddings(documents)


# --------------------------------------------------
# SEMANTIC SEARCH FUNCTION
# --------------------------------------------------

def semantic_search(query, top_k=3):

    query_embedding = get_embeddings([query])[0]

    # Cosine similarity
    similarities = np.dot(document_embeddings, query_embedding) / (
        np.linalg.norm(document_embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    top_index = top_indices[0]

    return {
        "category": categories[top_index],
        "response": documents[top_index],
        "similarity_score": float(similarities[top_index]),
        "results": [
            {
                "category": categories[i],
                "document": documents[i],
                "score": float(similarities[i])
            }
            for i in top_indices
        ]
    }


# --------------------------------------------------
# API REQUEST MODEL
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str


# --------------------------------------------------
# API ENDPOINT
# --------------------------------------------------

@app.post("/chat")
def chat(request: QueryRequest):

    query = request.query.strip()

    if not query:
        return {
            "category": "Unknown",
            "response": "Please enter a customer message."
        }

    result = semantic_search(query)

    return {
        "query": query,
        "category": result["category"],
        "response": result["response"],
        "similarity_score": round(result["similarity_score"], 4)
    }


# --------------------------------------------------
# CHATBOT FRONTEND
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html>
<head>
    <title>E-Commerce Support Chatbot</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f9;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }

        .chat-container {
            width: 90%;
            max-width: 600px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: #2563eb;
            color: white;
            padding: 20px;
            text-align: center;
        }

        .header h2 {
            margin: 0;
        }

        .header p {
            margin-bottom: 0;
        }

        .chat-box {
            height: 350px;
            overflow-y: auto;
            padding: 20px;
        }

        .message {
            padding: 12px;
            margin: 10px 0;
            border-radius: 8px;
            line-height: 1.5;
        }

        .user {
            background: #dbeafe;
            text-align: right;
        }

        .bot {
            background: #f1f5f9;
            text-align: left;
        }

        .input-area {
            display: flex;
            padding: 15px;
            border-top: 1px solid #ddd;
            gap: 10px;
        }

        input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 15px;
        }

        button {
            padding: 12px 20px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }
    </style>
</head>

<body>

<div class="chat-container">

    <div class="header">
        <h2>🛒 E-Commerce Support Chatbot</h2>
        <p>Ask about orders, returns, payments, or refunds</p>
    </div>

    <div class="chat-box" id="chatBox">
        <div class="message bot">
            Hello! How can I help you today?
        </div>
    </div>

    <div class="input-area">
        <input
            type="text"
            id="userInput"
            placeholder="Enter your customer message..."
            onkeydown="if(event.key === 'Enter') sendMessage()"
        >

        <button onclick="sendMessage()">Send</button>
    </div>

</div>

<script>

async function sendMessage() {

    const input = document.getElementById("userInput");
    const chatBox = document.getElementById("chatBox");

    const query = input.value.trim();

    if (!query) return;

    // Display user message
    chatBox.innerHTML += `
        <div class="message user">
            <b>You:</b> ${query}
        </div>
    `;

    input.value = "";

    // Display loading message
    chatBox.innerHTML += `
        <div class="message bot" id="loading">
            Thinking...
        </div>
    `;

    chatBox.scrollTop = chatBox.scrollHeight;

    try {

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: query
            })
        });

        const data = await response.json();

        document.getElementById("loading").remove();

        chatBox.innerHTML += `
            <div class="message bot">
                <b>Category:</b> ${data.category}<br><br>
                <b>Response:</b> ${data.response}<br><br>
                <small>Similarity Score: ${data.similarity_score}</small>
            </div>
        `;

    } catch (error) {

        document.getElementById("loading").innerHTML =
            "Sorry, something went wrong. Please try again.";

    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

</script>

</body>
</html>
"""
