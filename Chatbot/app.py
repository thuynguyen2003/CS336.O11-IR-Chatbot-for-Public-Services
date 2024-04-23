from flask import Flask, render_template, request, jsonify
from model import Chatbot
app = Flask(__name__)

@app.route("/")
def index():
    default_message = "Tôi là chatbot hỗ trợ giải đáp dịch vụ công, rất mong có thể giúp bạn giải đáp thắc mắc có liên quan."
    return render_template("chat.html", default_message=default_message)

@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):
    for step in range(10):
        answer = Chatbot(text)        
        return answer
    
if __name__ == '__main__':
    app.run()