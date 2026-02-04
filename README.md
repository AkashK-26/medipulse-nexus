# MediPulse Nexus - Hospital Analytics Dashboard

A comprehensive, real-time hospital operations analytics dashboard built with FastAPI, PostgreSQL, and Chart.js.

## Features

- Real-time KPI tracking (ALOS, Bed Occupancy, Readmission Rates)
- Predictive analytics with 7-day forecasting
- Interactive charts with dark mode UI
- Department-wise performance analysis
- Doctor utilization tracking
- Operational bottleneck detection

## Tech Stack

- **Backend:** FastAPI (Python)
- **Database:** PostgreSQL
- **Frontend:** HTML5, CSS3, JavaScript
- **Charts:** Chart.js with ChartDataLabels plugin
- **ML:** Facebook Prophet for forecasting

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Application
```bash
python main.py
```

Access at: http://localhost:8000

## License

MIT License
