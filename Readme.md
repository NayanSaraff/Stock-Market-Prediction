# Agent based Stock Market Prediction

This project is an agent-based stock market prediction application built with Python and Streamlit. It uses machine learning models (like LSTM) and various technical indicators to forecast stock prices.

## Installation Instructions

Follow these steps to set up and run the project locally.

### 1. Clone the repository
```bash
git clone https://github.com/NayanSaraff/Stock-Market-Prediction.git
cd Stock-Market-Prediction
```

### 2. Create a Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to manage dependencies.

**Using venv:**
```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
The dashboard is built using Streamlit. Start the local Streamlit server by running:
```bash
streamlit run app.py
```

This will automatically open the dashboard in your default web browser (typically at `http://localhost:8501`).
