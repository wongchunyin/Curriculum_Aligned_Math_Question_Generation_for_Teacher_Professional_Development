from flask import Flask
from pyngrok import ngrok
import threading

# Your existing app.py code here
# ... (copy all your functions and routes)

if __name__ == '__main__':
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)