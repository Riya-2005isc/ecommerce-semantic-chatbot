from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


# =====================================================
# KNOWLEDGE BASE
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
        "message": "E-commerce chatbot is running"
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

    query_vector = vectorizer.transform([user_message])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    best_index = similarities.argmax()
    best_score = float(similarities[best_index])

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
# PROFESSIONAL CHATBOT INTERFACE
# =====================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>ShopEase Customer Support</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: #f3f7fc;
            color: #17345f;
        }

        .app {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* ================= HEADER ================= */

        .header {
            height: 86px;
            background: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 5%;
            border-bottom: 1px solid #e6edf6;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            background: linear-gradient(135deg, #1976ed, #42a5f5);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 27px;
            box-shadow: 0 6px 14px rgba(25, 118, 237, 0.22);
        }

        .brand-name {
            font-size: 29px;
            font-weight: 800;
            letter-spacing: -1px;
            color: #18345e;
        }

        .brand-name span {
            color: #1976ed;
        }

        .brand-tagline {
            font-size: 12px;
            color: #7085a3;
            margin-top: 3px;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 28px;
        }

        .trust-message {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 13px;
            font-weight: 600;
            color: #58708e;
        }

        .trust-icon {
            font-size: 25px;
        }

        .user-icon {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: #eef5ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
        }

        /* ================= MAIN LAYOUT ================= */

        .main {
            flex: 1;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 25px;
            padding: 28px 5%;
            max-width: 1600px;
            width: 100%;
            margin: auto;
        }

        /* ================= SIDEBAR ================= */

        .sidebar {
            background: linear-gradient(150deg, #146bd5, #3289e9);
            border-radius: 22px;
            padding: 32px 25px 25px;
            color: white;
            position: relative;
            overflow: hidden;
            min-height: 650px;
        }

        .sidebar::before {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 50%;
            top: 70px;
            right: -100px;
        }

        .sidebar::after {
            content: "";
            position: absolute;
            width: 300px;
            height: 300px;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 50%;
            bottom: -170px;
            left: -120px;
        }

        .sidebar-content {
            position: relative;
            z-index: 2;
        }

        .sidebar h1 {
            font-size: 30px;
            line-height: 1.25;
            margin: 0 0 20px;
            color: white;
        }

        .sidebar-description {
            font-size: 16px;
            line-height: 1.8;
            color: #e8f2ff;
            margin-bottom: 20px;
        }

        .robot-area {
            text-align: center;
            margin: 15px 0 25px;
        }

        .robot {
            width: 145px;
            height: 110px;
            margin: auto;
            background: white;
            border-radius: 45% 45% 35% 35%;
            position: relative;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }

        .robot::before {
            content: "";
            position: absolute;
            width: 82px;
            height: 62px;
            background: #173d78;
            border-radius: 35px;
            left: 31px;
            top: 25px;
        }

        .robot::after {
            content: "•  •";
            white-space: pre;
            position: absolute;
            color: white;
            font-size: 25px;
            letter-spacing: 10px;
            left: 39px;
            top: 36px;
        }

        .robot-antenna {
            width: 5px;
            height: 22px;
            background: white;
            position: absolute;
            top: -20px;
            left: 70px;
        }

        .robot-antenna::before {
            content: "";
            width: 15px;
            height: 15px;
            background: #bde1ff;
            border-radius: 50%;
            position: absolute;
            top: -10px;
            left: -5px;
        }

        .speech-bubble {
            position: absolute;
            right: 18px;
            top: -8px;
            background: white;
            color: #1976ed;
            border-radius: 16px;
            padding: 13px;
            font-size: 22px;
            box-shadow: 0 7px 16px rgba(0,0,0,0.12);
        }

        /* ================= TOPICS ================= */

        .topics {
            background: rgba(255,255,255,0.97);
            border-radius: 18px;
            padding: 18px;
            color: #17345f;
        }

        .topics-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .topic {
            background: #f8fbff;
            border: 1px solid #edf2f8;
            border-radius: 13px;
            padding: 13px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            cursor: pointer;
            transition: 0.2s;
        }

        .topic:hover {
            background: #eaf4ff;
            transform: translateX(3px);
        }

        .topic-left {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 14px;
            font-weight: 600;
        }

        .topic-icon {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .blue {
            background: #e3f0ff;
        }

        .green {
            background: #e2f8ee;
        }

        .purple {
            background: #eee7ff;
        }

        .orange {
            background: #fff0dc;
        }

        .arrow {
            color: #7290b5;
            font-size: 20px;
        }

        /* ================= CHAT PANEL ================= */

        .chat-panel {
            background: white;
            border-radius: 22px;
            box-shadow: 0 8px 30px rgba(38, 80, 130, 0.08);
            display: flex;
            flex-direction: column;
            min-height: 650px;
            overflow: hidden;
        }

        .chat-header {
            padding: 25px 30px;
            border-bottom: 1px solid #edf1f6;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .chat-avatar {
            width: 55px;
            height: 55px;
            background: #e5f1ff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
        }

        .chat-title {
            font-size: 22px;
            font-weight: 700;
            color: #17345f;
        }

        .chat-status {
            color: #7c91ad;
            font-size: 14px;
            margin-top: 5px;
        }

        #messages {
            flex: 1;
            padding: 35px 30px;
            overflow-y: auto;
            min-height: 390px;
        }

        .welcome-message {
            display: flex;
            gap: 14px;
            align-items: flex-start;
        }

        .small-avatar {
            width: 42px;
            height: 42px;
            flex-shrink: 0;
            background: #e5f1ff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
        }

        .welcome-bubble {
            background: #eaf4ff;
            border-radius: 0 18px 18px 18px;
            padding: 18px 22px;
            color: #17345f;
            font-size: 16px;
            line-height: 1.7;
            max-width: 80%;
        }

        .time {
            font-size: 12px;
            color: #91a2b8;
            margin-top: 8px;
        }

        .message {
            margin: 18px 0;
            display: flex;
            flex-direction: column;
        }

        .message.user {
            align-items: flex-end;
        }

        .message.bot {
            align-items: flex-start;
        }

        .message-bubble {
            max-width: 80%;
            padding: 15px 19px;
            border-radius: 16px;
            font-size: 15px;
            line-height: 1.6;
        }

        .user .message-bubble {
            background: #1976ed;
            color: white;
            border-radius: 16px 16px 0 16px;
        }

        .bot .message-bubble {
            background: #eaf4ff;
            color: #17345f;
            border-radius: 0 16px 16px 16px;
        }

        .message-category {
            font-size: 12px;
            color: #8094ad;
            margin-top: 6px;
        }

        /* ================= INPUT ================= */

        .input-section {
            background: #f8fbff;
            border-top: 1px solid #edf1f6;
            padding: 25px 30px;
        }

        .input-area {
            display: flex;
            gap: 14px;
        }

        #message {
            flex: 1;
            border: 1px solid #d9e4f0;
            border-radius: 30px;
            padding: 17px 22px;
            font-size: 16px;
            outline: none;
            color: #17345f;
            background: white;
        }

        #message:focus {
            border-color: #1976ed;
            box-shadow: 0 0 0 3px rgba(25,118,237,0.08);
        }

        #sendButton {
            border: none;
            border-radius: 14px;
            background: #1976ed;
            color: white;
            font-size: 16px;
            font-weight: 700;
            padding: 0 28px;
            cursor: pointer;
            transition: 0.2s;
        }

        #sendButton:hover {
            background: #125dbd;
        }

        #sendButton:disabled {
            background: #9cbce1;
            cursor: not-allowed;
        }

        /* ================= FOOTER ================= */

        .footer {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 45px;
            padding: 18px 5%;
            color: #7890ad;
            font-size: 13px;
            background: white;
            border-top: 1px solid #e7eef6;
        }

        .footer-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .footer-icon {
            font-size: 20px;
        }

        /* ================= RESPONSIVE ================= */

        @media (max-width: 1000px) {
            .main {
                grid-template-columns: 1fr;
            }

            .sidebar {
                min-height: auto;
            }

            .robot-area {
                display: none;
            }

            .topics {
                margin-top: 20px;
            }

            .chat-panel {
                min-height: 600px;
            }
        }

        @media (max-width: 600px) {
            .header {
                padding: 0 20px;
            }

            .header-right {
                gap: 10px;
            }

            .trust-message {
                display: none;
            }

            .brand-name {
                font-size: 23px;
            }

            .main {
                padding: 15px;
            }

            .sidebar,
            .chat-panel {
                border-radius: 16px;
            }

            .sidebar h1 {
                font-size: 25px;
            }

            .chat-header,
            #messages,
            .input-section {
                padding-left: 18px;
                padding-right: 18px;
            }

            .input-area {
                gap: 8px;
            }

            #sendButton {
                padding: 0 18px;
            }

            .footer {
                flex-direction: column;
                gap: 12px;
            }
        }
    </style>
</head>


<body>

<div class="app">

    <!-- HEADER -->
    <header class="header">

        <div class="brand">
            <div class="brand-logo">🛒</div>

            <div>
                <div class="brand-name">
                    Shop<span>Ease</span>
                </div>

                <div class="brand-tagline">
                    Shop More. Worry Less.
                </div>
            </div>
        </div>

        <div class="header-right">

            <div class="trust-message">
                <span class="trust-icon">🛡️</span>
                <span>Your Satisfaction<br>Our Priority</span>
            </div>

            <div class="user-icon">👤</div>

        </div>

    </header>


    <!-- MAIN -->
    <main class="main">

        <!-- SIDEBAR -->
        <aside class="sidebar">

            <div class="sidebar-content">

                <h1>
                    Hi! I'm your<br>
                    E-Commerce<br>
                    Support Assistant 👋
                </h1>

                <p class="sidebar-description">
                    I'm here to help you with your orders,
                    returns, payments and refunds.
                    Just type your question below!
                </p>

                <div class="robot-area">
                    <div class="robot">
                        <div class="robot-antenna"></div>
                        <div class="speech-bubble">💬</div>
                    </div>
                </div>

                <div class="topics">

                    <div class="topics-title">
                        💬 &nbsp; Common Topics
                    </div>

                    <div class="topic" onclick="useTopic('Where is my order?')">
                        <div class="topic-left">
                            <div class="topic-icon blue">🚚</div>
                            Order Tracking
                        </div>
                        <div class="arrow">›</div>
                    </div>

                    <div class="topic" onclick="useTopic('I want to return my product.')">
                        <div class="topic-left">
                            <div class="topic-icon green">📦</div>
                            Product Returns
                        </div>
                        <div class="arrow">›</div>
                    </div>

                    <div class="topic" onclick="useTopic('My payment failed.')">
                        <div class="topic-left">
                            <div class="topic-icon purple">💳</div>
                            Payment Issues
                        </div>
                        <div class="arrow">›</div>
                    </div>

                    <div class="topic" onclick="useTopic('Where is my refund?')">
                        <div class="topic-left">
                            <div class="topic-icon orange">↻</div>
                            Refund Status
                        </div>
                        <div class="arrow">›</div>
                    </div>

                </div>

            </div>

        </aside>


        <!-- CHAT PANEL -->
        <section class="chat-panel">

            <div class="chat-header">

                <div class="chat-avatar">💬</div>

                <div>
                    <div class="chat-title">Chat Support</div>
                    <div class="chat-status">● Typically replies instantly</div>
                </div>

            </div>


            <div id="messages">

                <div class="welcome-message">

                    <div class="small-avatar">🤖</div>

                    <div>
                        <div class="welcome-bubble">
                            <strong>Hello! 👋</strong><br>
                            I'm your e-commerce support assistant.
                            How can I help you today?
                        </div>

                        <div class="time">Just now</div>
                    </div>

                </div>

            </div>


            <div class="input-section">

                <div class="input-area">

                    <input
                        type="text"
                        id="message"
                        placeholder="Type your question..."
                        onkeydown="if(event.key === 'Enter') sendMessage()"
                    >

                    <button id="sendButton" onclick="sendMessage()">
                        ➤ &nbsp; Send
                    </button>

                </div>

            </div>

        </section>

    </main>


    <!-- FOOTER -->
    <footer class="footer">

        <div class="footer-item">
            <span class="footer-icon">🛡️</span>
            Secure & Trusted
        </div>

        <div class="footer-item">
            <span class="footer-icon">🚚</span>
            Fast & Reliable Delivery
        </div>

        <div class="footer-item">
            <span class="footer-icon">♡</span>
            We're Always Here for You
        </div>

    </footer>

</div>


<script>

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }


    function getCurrentTime() {
        return new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
        });
    }


    function useTopic(topic) {
        document.getElementById("message").value = topic;
        document.getElementById("message").focus();
    }


    async function sendMessage() {

        const input = document.getElementById("message");
        const messages = document.getElementById("messages");
        const sendButton = document.getElementById("sendButton");

        const message = input.value.trim();

        if (!message) {
            return;
        }

        // Display user message
        messages.innerHTML += `
            <div class="message user">
                <div class="message-bubble">
                    ${escapeHtml(message)}
                </div>
                <div class="message-category">
                    ${getCurrentTime()}
                </div>
            </div>
        `;

        input.value = "";
        sendButton.disabled = true;
        sendButton.innerHTML = "Sending...";

        messages.scrollTop = messages.scrollHeight;

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
                    <div class="message-bubble">
                        ${escapeHtml(data.response)}
                    </div>

                    <div class="message-category">
                        Category: ${escapeHtml(data.category)}
                        <br>
                        Similarity Score: ${data.similarity_score}
                    </div>
                </div>
            `;

        } catch (error) {

            messages.innerHTML += `
                <div class="message bot">
                    <div class="message-bubble">
                        Sorry, something went wrong. Please try again.
                    </div>
                </div>
            `;

        }

        sendButton.disabled = false;
        sendButton.innerHTML = "➤ &nbsp; Send";

        messages.scrollTop = messages.scrollHeight;
    }

</script>

</body>
</html>
"""
