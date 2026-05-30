# Quick Colab setup - copy this into a Colab cell

# Install packages
!pip install flask pyngrok transformers torch

# Setup ngrok
import os
from pyngrok import ngrok
ngrok.set_auth_token(os.environ["NGROK_AUTH_TOKEN"])

# Add this to the end of your app.py
if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"🌐 Your app is live at: {public_url}")
    app.run(host='0.0.0.0', port=5000, use_reloader=False)
