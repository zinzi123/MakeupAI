from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Fetch the Google API key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize Google Gemini Flash model
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.5,
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    print(f"Error initializing ChatGoogleGenerativeAI: {e}")

# Sample product database and follow-up questions
products = {
    "skincare": [
        {"name": "CeraVe Moisturizing Cream", "description": "Rich in ceramides, helps retain moisture.", "ingredients": ["Ceramides", "Hyaluronic Acid"]},
        {"name": "La Roche-Posay Moisturizer", "description": "Great for sensitive and dry skin.", "ingredients": ["Ceramides", "Niacinamide"]},
        {"name": "Neutrogena Hydro Boost Gel-Cream", "description": "Intense hydration gel formula.", "ingredients": ["Hyaluronic Acid", "Glycerin"]},
    ]
}

follow_up_questions = {
    "skincare": [
        "Do you have any specific concerns like dryness or allergies?",
        "What is your skin tone? (e.g., fair, medium, dark)",
        "Do you prefer products with natural ingredients?",
        "Are you looking for a day or night cream?",
        "Do you need products with SPF protection?"
    ]
}

# Database setup
def init_db():
    with sqlite3.connect('chatbot.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                skin_type TEXT,
                preferences TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS follow_up_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                question TEXT,
                answer TEXT,
                FOREIGN KEY(user_id) REFERENCES user_profiles(id)
            )
        ''')
        conn.commit()

# Save user profile to database
def save_user_profile(name, skin_type, preferences):
    with sqlite3.connect('chatbot.db') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO user_profiles (name, skin_type, preferences) VALUES (?, ?, ?)", 
                       (name, skin_type, preferences))
        conn.commit()

# Save follow-up answers
def save_follow_up_response(user_id, question, answer):
    with sqlite3.connect('chatbot.db') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO follow_up_responses (user_id, question, answer) VALUES (?, ?, ?)",
                       (user_id, question, answer))
        conn.commit()

# Get user profile from database
def get_user_profile(name):
    with sqlite3.connect('chatbot.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_profiles WHERE name = ?", (name,))
        return cursor.fetchone()

# Get follow-up questions for a category
def get_follow_up_questions(category):
    return follow_up_questions.get(category, [])

# Get product recommendations for a category
def get_product_recommendations(category):
    return products.get(category, [])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')

    # Handle user input for skincare category and follow-up questions
    name = "John Doe"  # For example, replace this with dynamic user input handling
    category = "skincare"  # Assume skincare for now
    user_profile = get_user_profile(name)

    if user_profile:
        # User already exists, so get recommendations and follow-up
        user_id = user_profile[0]
        recommendations = get_product_recommendations(category)
        follow_ups = get_follow_up_questions(category)
        
        # Store follow-up answers (here we simply simulate answers)
        save_follow_up_response(user_id, follow_ups[0], "Dry Skin")
        
        # Respond with product recommendations and questions
        response = f"We recommend the following products: {', '.join([p['name'] for p in recommendations])}"
        response += f" {follow_ups[0]}"
    else:
        # New user, prompt for profile details
        save_user_profile(name, "Dry", category)
        response = "Welcome! Please answer some follow-up questions."

    return jsonify({"response": response})

if __name__ == '__main__':
    init_db()  # Ensure DB is initialized
    app.run(debug=True)
