from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


# Knowledge base
documents = [
    "Where is my order?",
    "How can I track my order?",
    "I want to know the delivery status of my package.",
    "When will my order arrive?",

    "I want to return my product.",
    "How can I send back an item?",
    "I received a damaged product and want to return it.",
    "What is the process for returning a product?",

    "My payment failed.",
    "My payment was unsuccessful.",
    "I could not complete the payment.",
    "The transaction failed while placing my order.",

    "Where is my refund?",
    "When will I receive my refund?",
    "My refund has not arrived.",
    "How can I check my refund status?"
]

categories = [
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",

    "Product Return",
    "Product Return",
    "Product Return",
    "Product Return",

    "Payment Failure",
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",

    "Refund Status",
    "Refund Status",
    "Refund Status",
    "Refund Status"
]

responses = {
    "Order Tracking":
        "You can track your order using the tracking link provided in your order confirmation email.",

    "Product Return":
        "You can request a return from the Orders section. Select the product and choose the Return option.",

    "Payment Failure":
        "Please check your internet connection, payment details, or try another payment method.",

    "Refund Status":
        "Refunds are generally processed after the returned product is verified. Please check your order details for the latest update."
}


# Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

document_vectors = vectorizer.fit_transform(documents)


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>E-Commerce Chatbot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f2f4f7;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }

            .chatbox {
                width: 420px;
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            }

            h2 {
                text-align: center;
                color: #333;
            }

            #messages {
                height: 300px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 12px;
                margin-bottom: 15px;
            }

            .user {
                text-align: right;
                color: #155724;
                margin: 10px;
            }

            .bot {
                text-align: left;
                color: #004085;
                margin: 10px;
            }

            input {
                width: 70%;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }

            button {
                padding: 10px 15px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
        </style>
    </head>

    <body>
        <div class="chatbox">
            <h2>E-Commerce Chatbot</h2>

            <div id="messages"></div>

            <input
                id="message"
                type="text"
                placeholder="Type your question..."
                onkeydown="if(event.key === 'Enter') sendMessage()"
            >

            <button onclick="sendMessage()">Send</button>
        </div>

        <script>
            async function sendMessage() {
                const input = document.getElementById("message");
                const message = input.value.trim();

                if (!message) return;

                const messages = document.getElementById("messages");

                messages.innerHTML +=
                    `<div class="user"><b>You:</b> ${message}</div>`;

                input.value = "";

                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        message: message
                    })
                });

                const data = await response.json();

                messages.innerHTML +=
                    `<div class="bot">
                        <b>Bot:</b> ${data.response}<br>
                        <small>Category: ${data.category}</small>
                    </div>`;

                messages.scrollTop = messages.scrollHeight;
            }
        </script>
    </body>
    </html>
    """


@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.strip()

    if not user_message:
        return {
            "category": "Unknown",
            "response": "Please enter a message."
        }

    # Convert user query into TF-IDF vector
    query_vector = vectorizer.transform([user_message])

    # Calculate similarity with all documents
    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    # If similarity is too low, avoid incorrect classification
    if best_score < 0.15:
        return {
            "category": "Unknown",
            "response":
                "Sorry, I could not understand your request. "
                "Please ask about order tracking, product returns, payment failure, or refund status."
        }

    category = categories[best_index]

    return {
        "category": category,
        "response": responses[category],
        "similarity_score": round(float(best_score), 3)
    }
