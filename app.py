from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
from chatbot.chatbot import chatbot_response, process_input
import webbrowser
import threading

# ✅ Initialize Flask App
app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
app.secret_key = "temp_secret_key"  # Required for session handling

# ✅ Enable CORS for handling cross-domain requests
CORS(app)

# ✅ Initialize state variables
step = 0
user_data = {}
session_active = True

# ✅ Temporary user storage (since we aren't using a database)
users = {
    "testuser": {"password": "password123"}
}

# ✅ Redirect to login if not authenticated
@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# ✅ Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # ✅ Check if user exists and password is correct
        if username in users and users[username]["password"] == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="❌ Invalid username or password. Try again.")

    return render_template("login.html")

# ✅ Register Page
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # ✅ Check if username already exists
        if username in users:
            return render_template("register.html", error="⚠ Username already taken. Choose another.")

        # ✅ Register user
        users[username] = {"password": password}

        # ✅ Redirect user to the login page after successful registration
        return redirect(url_for("login"))

    return render_template("register.html")

# ✅ Dashboard (only accessible after login)
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))  # Redirect to login if not authenticated

    username = session["user"]
    return render_template("index.html", username=username)

# ✅ Chatbot Response Handling
@app.route("/get_response", methods=["POST"])
def get_response():
    global step, user_data, session_active

    try:
        user_message = request.json.get("message", "").strip()

        if not user_message:
            return jsonify({"response": "⚠ Please type a message."}), 400

        # ✅ Prevent further input if session is ended
        if not session_active:
            return jsonify({
                "response": "✅ Session has ended. Please refresh the page to start a new diagnosis."
            })

        # ✅ Start new session if step = 0
        if step == 0 and session_active:
            user_data = {}

        # ✅ Get chatbot response
        bot_response = chatbot_response(user_message, step, user_data)

        # ✅ Step Management
        if step < 10:
            step += 1
        elif step == 10:
            # ✅ Predict disease at step 10 (AFTER breathing issue input)
            step = 11  # Move to restart/exit prompt
        elif step == 11:
            user_input = process_input(user_message).lower()
            if user_input == "yes":
                step = 0  # Restart diagnosis
            else:
                # ✅ End session and reset state (but don't restart)
                step = 0
                user_data = {}
                session_active = False
                return jsonify({
                    "response": "✅ Thank you for using the AI Medical Assistant. Session has ended. Please refresh the page to start a new session."
                })

        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"response": "⚠ An error occurred. Please try again later."}), 500

# ✅ Logout Route
@app.route('/logout', methods=["POST"])
def logout():
    """
    Handles user logout and redirects to the home page.
    """
    global user_data, step, session_active
    user_data = {}
    step = 0
    session_active = True
    session.pop("user", None)
    return redirect(url_for("login"))

# ✅ Auto Open Browser on Launch
def open_browser():
    """
    Open the default web browser after Flask starts.
    """
    webbrowser.open("http://127.0.0.1:8000/")

# ✅ Start Flask App
if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()  # Open browser after 1.5 seconds
    app.run(debug=True, port=8000)


