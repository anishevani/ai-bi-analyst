# AI BI Analyst

Ask plain-English questions about sales (or any tabular) data and get answers with charts.

## Features

- Sample 2024 sales dataset (`sample_data/sales_data.csv`)
- Upload your own CSV
- Natural-language questions (trend, region, product, category, segment, reps, volume)
- **Local analyst** works offline with no API key
- Optional **OpenAI** mode when `OPENAI_API_KEY` is set in `.env`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy or edit `.env`:

```
OPENAI_API_KEY=          # optional
OPENAI_MODEL=gpt-4o-mini
```

## Run

```bash
streamlit run app.py --server.port 4521 --server.address 0.0.0.0
```

Open [http://127.0.0.1:4521](http://127.0.0.1:4521).

## Project layout

```
ai-bi-analyst/
├── app.py
├── requirements.txt
├── .env
├── sample_data/
│   └── sales_data.csv
└── README.md
```

## Example questions

- What is the revenue trend over time?
- Which region performs best?
- What are the top products by revenue?
- How does revenue break down by category?
- Which sales rep leads?
