# 🐦 Tweet Foul Language Classifier

Binary classification API for detecting foul/offensive tweets using SVM with **96.65% F1 score**.

---

## 🚀 Quick Start

### Local
```bash
git clone <repo-url> && cd tweet-classifier
pip install -r requirements.txt
python app.py
```
**Access:** http://localhost:8000/docs

### Docker
```bash
docker build -t tweet-classifier .
docker run -p 8000:8000 tweet-classifier
```
**Access:** http://localhost:8000/docs

---

## 📋 Prerequisites

- Python 3.8+
- pip
- Docker (optional)

---

## 💻 Local Setup

### 1. Clone & Navigate
```bash
git clone <repo-url>
cd tweet-classifier
```

### 2. Virtual Environment (Optional)
```bash
# Create
python -m venv venv

# Activate
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Model Files
```bash
ls *.pkl
# Expected: model.pkl, vectorizer.pkl, threshold.pkl, model_metadata.pkl
```

If missing, train models:
```bash
jupyter notebook train_models.ipynb  # Run all cells
```

### 5. Start Server
```bash
# Method 1
python app.py

# Method 2
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access API
- **Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health
- **Metrics:** http://localhost:8000/metrics

---

## 🐳 Docker Setup

### 1. Build Image
```bash
docker build -t tweet-classifier .
```

### 2. Run Container
```bash
# Foreground
docker run -p 8000:8000 tweet-classifier

# Background
docker run -d -p 8000:8000 --name tweet-api tweet-classifier
```

### 3. Manage Container
```bash
docker ps                    # View running
docker logs tweet-api        # View logs
docker stop tweet-api        # Stop
docker start tweet-api       # Start
docker rm tweet-api          # Remove
```

---

## 🧪 Testing

### API Testing

**Browser:** http://localhost:8000/docs → Try `/predict` endpoint

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this day!"}'
```

**Python:**
```python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "I love this day!"}
)
print(response.json())
```

### Unit Tests
```bash
pytest tests/test_api.py -v
```

---

## 📊 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/health` | GET | Detailed status |
| `/metrics` | GET | Model performance |
| `/predict` | POST | Classify tweet |

### Example Request
```json
{
  "text": "I love this beautiful day!"
}
```

### Example Response
```json
{
  "text": "I love this beautiful day!",
  "prediction": "proper",
  "label": 0,
  "confidence": 0.3479,
  "threshold": 0.7804
}
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| model.pkl not found | Run training notebook |
| Port 8000 in use | `uvicorn app:app --port 8001` |
| Docker daemon error | Start Docker Desktop |
| Permission denied (venv) | `Set-ExecutionPolicy RemoteSigned` (Windows) |

---

## 📈 Performance

- **Accuracy:** 94.53%
- **Precision:** 98.59%
- **Recall:** 94.79%
- **F1 Score:** 96.65%

---

## 🏗️ Architecture

```
Tweet Input → Preprocessing → TF-IDF (5000 features) → SVM → Classification
```

### Tech Stack
- **ML:** scikit-learn, pandas, numpy
- **API:** FastAPI, uvicorn, pydantic
- **Deployment:** Docker

---

## 📂 Project Structure

```
tweet-classifier/
├── app.py                  # FastAPI application
├── train_models.ipynb      # Model training
├── model.pkl              # Trained SVM
├── vectorizer.pkl         # TF-IDF vectorizer
├── threshold.pkl          # Decision threshold
├── requirements.txt       # Dependencies
├── Dockerfile            # Container config
├── tests/
│   └── test_api.py       # API tests
└── README.md            # This file
```

---

## 🔑 Key Commands

```bash
# Local
python app.py                          # Start API
pytest tests/test_api.py -v           # Run tests
deactivate                             # Exit venv

# Docker
docker build -t tweet-classifier .     # Build
docker run -p 8000:8000 tweet-classifier  # Run
docker ps                              # List containers
docker stop tweet-api                  # Stop
```

---

## ✅ Success Checklist

- [ ] API starts without errors
- [ ] http://localhost:8000/docs loads
- [ ] `/predict` returns valid predictions
- [ ] All tests pass
- [ ] Docker build succeeds (optional)

---

## 📞 Support

For issues:
1. Check troubleshooting table
2. Verify all model files exist
3. Check logs: `docker logs tweet-api`

---

**Built with 96.65% F1 Score | Production-Ready | Fully Tested**