# James Callahan Portfolio (Dash)

This is the Dash version of the portfolio (the original Streamlit app is in `ML_Docker_Azure_webapp/`).

## Setup

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your resume**: Place your resume PDF in the `assets` folder as `resume.pdf`:
   ```bash
   cp /path/to/your/resume.pdf portfolio_dash/assets/resume.pdf
   ```
   The home page will display and offer a download for this file.

4. **Toastmasters page (optional)**: To use the Table Topics generator, create a `.env` file in `portfolio_dash/` with your Azure OpenAI settings:
   ```
   API_KEY=your_azure_openai_key
   endpoint=https://your-resource.openai.azure.com/
   deployment_name=your_deployment_name
   ```
   Copy `.env.example` to `.env` and fill in the values.

## Run the app

From the `portfolio_dash` directory:

```bash
python app.py
```

Then open http://127.0.0.1:8050 in your browser.

## Pages

- **Home** – Intro, photo, resume viewer/download, LinkedIn & GitHub links
- **Clustering** – Hierarchical clustering on US crash data (state, sample size, distance threshold)
- **Toastmasters** – Table Topics question generator (Azure OpenAI)
- **Car Prices** – Embedded RShiny app (Shinyapps)

## Data

`crash_data_prepped.csv` is included for the Clustering page. It was copied from the original Streamlit app’s data prep; the source is [US Accidents (Kaggle)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data).
