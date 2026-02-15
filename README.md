# ML Backend Engine - Kanini Triage System

FastAPI backend for ML-based patient triage predictions using XGBoost and SHAP explainability.

## Features

- **ML Predictions**: Risk level, department recommendations, and explainability
- **Supabase Integration**: Fetches patient data from Supabase
- **RESTful API**: Simple POST endpoint for predictions
- **Realistic Model**: ~71% accuracy, non-overfitted

## API Endpoint

### POST `/api/v1/process_visit`

**Request:**
```json
{
  "visit_id": 123
}
```

**Response:**
```json
{
  "visit_id": 123,
  "risk_level": "High",
  "risk_score": 0.92,
  "recommended_department": "Emergency",
  "department_scores": {
    "Emergency": 0.66,
    "Cardiology": 0.21,
    ...
  },
  "explainability": {
    "chest_pain_severity": 3.08,
    "bp_diastolic": 1.44,
    ...
  }
}
```

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment:**
Create `.env` file:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
```

3. **Train models:**
```bash
python scripts/train_models.py
```

4. **Run server:**
```bash
uvicorn app.main:app --reload --port 8000
```

## Deployment

Configured for Vercel deployment. See `vercel.json` for configuration.

## Project Structure

```
backend-engine/
├── app/
│   ├── api/v1/ml.py       # API endpoints
│   ├── core/
│   │   ├── config.py      # Configuration
│   │   └── database.py    # Supabase client
│   ├── models/
│   │   └── ml_engine.py   # ML prediction logic
│   └── main.py            # FastAPI app
├── scripts/
│   ├── train_models.py    # Model training
│   └── backfill_predictions.py  # One-time backfill
└── data/
    └── train.csv          # Training data
```

## License

MIT
