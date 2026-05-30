// Frontend configuration for different environments

const API_CONFIG = {
  // For local development
  LOCAL: 'http://localhost:5000',
  
  // For Colab - replace with your actual ngrok URL
  COLAB: 'https://YOUR_NGROK_URL_HERE.ngrok.io'
};

// Use this in your frontend fetch calls
const API_BASE_URL = API_CONFIG.COLAB; // Switch between LOCAL and COLAB

// Example usage:
// fetch(`${API_BASE_URL}/assess`, {
//   method: 'POST',
//   headers: { 'Content-Type': 'application/json' },
//   body: JSON.stringify(data)
// })