from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


# =====================================================
# E-COMMERCE KNOWLEDGE BASE
# =====================================================

documents = [
    # Order Tracking
    "Where is my order?",
    "How can I track my order?",
    "I want to track my package.",
    "What is the status of my order?",
    "Where is my package?",
    "When will my order arrive?",
    "When will my package be delivered?",
    "Tell me my delivery status.",
    "Can I get my order tracking information?",
    "My order has not arrived yet.",

    # Product Return
    "I want to return my product.",
    "How can I return an item?",
    "I want to send back the product I purchased.",
    "How do I return a damaged product?",
    "What is the process for returning a product?",
    "I received the wrong item and want to return it.",
    "Can I return my order?",
    "I need to request a product return.",
    "How can I send back an item?",
    "I am not satisfied with my product and want to return it.",

    # Payment Failure
    "My payment failed.",
    "My payment was unsuccessful.",
    "I could not complete the payment.",
    "The transaction failed while placing my order.",
    "Why did my payment fail?",
    "My card payment was declined.",
    "The payment did not go through.",
    "I received a payment failure message.",
    "The payment failed but the amount was deducted from my account.",
    "Money was deducted but my order was not placed.",
    "The amount was debited but the payment failed.",

    # Refund Status
    "Where is my refund?",
    "When will I receive my refund?",
    "My refund has not arrived.",
    "How can I check my refund status?",
    "I have not received my refund yet.",
    "When will the returned amount be credited?",
    "My refund is pending.",
    "The refund has not been credited to my account.",
    "I returned my product but have not received the money."
]


categories = [
    # Order Tracking
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",
    "Order Tracking",

    # Product Return
    "Product Return",
    "Product Return",
    "Product Return",
    "Product Return",
    "Product Return",
    "Product Return",
    "Product Return",
    "Product Return",
    "Product Return",
    "Product Return",

    # Payment Failure
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",
    "Payment Failure",

    # Refund Status
    "Refund Status",
    "Refund Status",
    "Refund Status",
    "Refund Status",
    "Refund Status",
    "Refund Status",
    "Refund Status",
    "Refund Status",
    "Refund Status"
]


# =====================================================
# BOT RESPONSES
# =====================================================

responses = {
    "Order Tracking":
        "You can track your order using the tracking link provided in your order confirmation email. "
        "You can also check the Orders section of your account.",

    "Product Return":
        "You can request a return from the Orders section. "
        "Select the product and choose the Return option. "
        "Follow the instructions to complete your return request.",

    "Payment Failure":
        "If the amount was deducted but the payment failed, please do not make another payment immediately. "
        "Check your bank or payment app for the transaction status. "
        "If the amount is not automatically refunded, contact customer support with your transaction ID.",

    "Refund Status":
        "Refunds are generally processed after the returned product is verified. "
        "Please check your order details for the latest refund update. "
        "If the refund is delayed, contact customer support with your order ID."
}


# =====================================================
# TF-IDF MODEL
# =====================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

document_vectors = vectorizer.fit_transform(documents)


# =====================================================
# REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):
    message: str


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "working",
        "message": "E-commerce chatbot API is running"
    }


# =====================================================
# CHAT API
# =====================================================

@app.post("/chat")
def chat(request: ChatRequest):

    user_message = request.message.strip()

    if not user_message:
        return {
            "category": "Unknown",
            "response": "Please enter a message.",
            "similarity_score": 0
        }

    # Convert user message into TF-IDF vector
    query_vector = vectorizer.transform([user_message])

    # Calculate similarity with all knowledge-base documents
    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    # Get the most similar document
    best_index = similarities.argmax()
    best_score = float(similarities[best_index])

    # Prevent unrelated questions from being classified incorrectly
    if best_score < 0.12:
        return {
            "category": "Unknown",
            "response":
                "Sorry, I could not understand your request. "
                "Please ask about order tracking, product returns, "
                "payment failure, or refund status.",
            "similarity_score": round(best_score, 3)
        }

    category = categories[best_index]

    return {
        "category": category,
        "response": responses[category],
        "similarity_score": round(best_score, 3)
    }


# =====================================================
# CHATBOT WEBPAGE
# =====================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>
    <html lang="en">

    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <title>E-Commerce Chatbot</title>

        <style>
            * {
                box-sizing: border-box;
            }

            body {
                font-family: Arial, sans-serif;
                background: #f1f3f6;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                padding: 20px;
            }

            .chat-container {
                width: 100%;
                max-width: 590px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 5px 25px rgba(0, 0, 0, 0.12);
                padding: 32px;
            }

            h1 {
                text-align: center;
                color: #26364a;
                margin-top: 0;
                margin-bottom: 25px;
                font-size: 30px;
            }

            #messages {
                height: 405px;
                overflow-y: auto;
                border: 1px solid #d7d7d7;
                border-radius: 8px;
                padding: 20px;
                background: #ffffff;
                margin-bottom: 18px;
            }

            .message {
                margin-bottom: 20px;
                line-height: 1.45;
                font-size: 18px;
            }

            .user {
                color: #006b38;
                text-align: right;
            }

            .bot {
                color: #0051a8;
                text-align: left;
            }

            .category {
                font-size: 15px;
                margin-top: 5px;
                color: #555;
            }

            .score {
                font-size: 13px;
                color: #777;
            }

            .input-area {
                display: flex;
                gap: 10px;
            }

            #message {
                flex: 1;
                padding: 14px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 7px;
                outline: none;
            }

            #message:focus {
                border-color: #1976d2;
            }

            button {
                padding: 14px 22px;
                font-size: 16px;
                color: white;
                background: #087cf5;
                border: none;
                border-radius: 7px;
                cursor: pointer;
            }

            button:hover {
                background: #0565ca;
            }

            button:disabled {
                background: #999;
                cursor: not-allowed;
            }

            .welcome {
                color: #555;
                text-align: center;
                font-size: 16px;
                margin-top: 10px;
            }
        </style>
    </head>

    <body>

        <div class="chat-container">

            <h1>E-Commerce Chatbot</h1>

            <div id="messages">
                <div class="welcome">
                    Ask me about order tracking, product returns,
                    payment failures, or refunds.
                </div>
            </div>

            <div class="input-area">

                <input
                    type="text"
                    id="message"
                    placeholder="Type your question..."
                    onkeydown="if (event.key === 'Enter') sendMessage()"
                >

                <button id="sendButton" onclick="sendMessage()">
                    Send
                </button>

            </div>

        </div>


        <script>

            async function sendMessage() {

                const input = document.getElementById("message");
                const sendButton = document.getElementById("sendButton");
                const messages = document.getElementById("messages");

                const message = input.value.trim();

                if (!message) {
                    return;
                }

                // Display user message
                messages.innerHTML += `
                    <div class="message user">
                        <b>You:</b> ${escapeHtml(message)}
                    </div>
                `;

                input.value = "";
                sendButton.disabled = true;
                sendButton.innerText = "Sending...";

                try {

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

                    messages.innerHTML += `
                        <div class="message bot">
                            <b>Bot:</b> ${escapeHtml(data.response)}
                            <div class="category">
                                Category: ${escapeHtml(data.category)}
                            </div>
                            <div class="score">
                                Similarity Score: ${data.similarity_score}
                            </div>
                        </div>
                    `;

                } catch (error) {

                    messages.innerHTML += `
                        <div class="message bot">
                            <b>Bot:</b>
                            Sorry, something went wrong. Please try again.
                        </div>
                    `;

                }

                sendButton.disabled = false;
                sendButton.innerText = "Send";

                messages.scrollTop = messages.scrollHeight;
            }


            function escapeHtml(text) {

                const div = document.createElement("div");
                div.textContent = text;
                return div.innerHTML;

            }

        </script>

    </body>

    </html>
    """


# This line is not required by Vercel,
# but the application can also run locally using:
#
# uvicorn api.index:app --reload
