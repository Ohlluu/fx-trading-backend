# FX Trading Backend API

FastAPI backend for Multi-Pair Forex Trading System with Smart Confluence analysis.

## Features

- Dual-pair trading system (XAUUSD + GBPUSD)
- Smart Confluence System with 60%+ win rate (backtested)
- Real-time price data from multiple sources
- Automated signal generation
- RESTful API endpoints

## Tech Stack

- Python 3.11
- FastAPI
- Pandas & NumPy for data analysis
- APScheduler for background tasks
- HTTPX for async HTTP requests

## Setup

1. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. Run server:
```bash
uvicorn app.multi_pair_main:app --reload --host 0.0.0.0 --port 8002
```

## API Endpoints

### Multi-Pair
- `GET /api/multi-pair/analysis` - Get analysis for both pairs

### XAUUSD (Gold)
- `GET /api/xauusd/analysis` - Get cached XAUUSD analysis
- `POST /api/xauusd/scan` - Force fresh XAUUSD scan

### GBPUSD
- `GET /api/gbpusd/analysis` - Get cached GBPUSD analysis
- `POST /api/gbpusd/scan` - Force fresh GBPUSD scan

## Deployment

### Railway (Recommended)

1. Push code to GitHub
2. Create new project in Railway
3. Connect GitHub repository
4. Add environment variables (API keys)
5. Deploy

The app will automatically use the Procfile for deployment.

## Environment Variables

- `TWELVE_DATA_API_KEY`: TwelveData API key for market data
- `OANDA_API_KEY`: OANDA API key (optional)
- `OANDA_ACCOUNT_ID`: OANDA account ID (optional)
- `ENVIRONMENT`: Set to "production" for production deployment

## License

Private - All Rights Reserved
