from flask import Flask, render_template, request, jsonify
import requests
import logging

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        # Send request to FastAPI backend
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={'question': question},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'answer': result['answer'],
                'retrieved_chunks': result.get('retrieved_chunks', [])
            })
        else:
            return jsonify({'error': 'Backend service error'}), 500
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return jsonify({'error': 'Unable to connect to backend service'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/health')
def health_check():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return jsonify({'status': 'healthy', 'backend': 'connected'})
        else:
            return jsonify({'status': 'unhealthy', 'backend': 'disconnected'}), 503
    except:
        return jsonify({'status': 'unhealthy', 'backend': 'disconnected'}), 503

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)