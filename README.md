# HypothesisAI

A hypothesis validation tool that uses statistical analysis and AI for result interpretation.

## Features
- **Statistical Analysis**: Regression, Correlation, P-Value, Confidence Intervals.
- **AI Interpretation**: Automated explanation of statistical results using OpenAI.
- **Research Evidence**: Fetches relevant papers from arXiv.
- **Multi-Dataset**: Supports Math and Portuguese student performance datasets.
- - **Interactive UI**: Interactive dashboard for running experiments.


## Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`

## Docker Deployment (Recommended)

This project includes full Docker support for easy deployment.

### 1. Build the Image
```bash
docker build -t hypothesis-ai .
```

### 2. Run the Container
You need to provide your OpenAI API key as an environment variable.

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY="your-key-here" hypothesis-ai
```

The app will be available at `http://localhost:8501`.

## Environment Variables

- `OPENAI_API_KEY`: Required for the AI Explanation layer.
