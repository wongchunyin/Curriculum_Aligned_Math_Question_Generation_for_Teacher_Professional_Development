# Math Skills Assessment

## Project Structure
```
website/
├── backend/          # Flask API server
│   ├── app.py
│   └── requirements.txt
├── frontend/         # React + MUI interface
│   ├── src/
│   ├── package.json
│   ├── vite.config.js
│   └── index_react.html
├── index.html        # Original simple version
├── script.js         # Original simple version
└── style.css         # Original simple version
```

## Running the Application

### Backend (Flask API)
```bash
cd backend
pip install -r requirements.txt
python app.py
# Runs on: http://127.0.0.1:5000
```

### Frontend (React + MUI)
```bash
cd frontend
npm install
npm run dev
# Runs on: http://localhost:3000
```

### Original Version (Simple)
```bash
python backend/app.py
# Visit: http://127.0.0.1:5000 (serves original index.html)
```