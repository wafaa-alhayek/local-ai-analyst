# Setup guide

This takes about 15 minutes, and most of that is downloading the model (about 4.7 GB) once.

## 1. Install Ollama (runs the AI model)

- **Windows / Mac:** download it from https://ollama.com/download and install it like any app.
- **Linux:** follow the instructions at https://ollama.com/download.

Then open a terminal and download the model:

```bash
ollama pull qwen2.5-coder:7b
```

On a laptop with 8 GB RAM, use the smaller model instead: `ollama pull qwen2.5:3b`.

Ollama keeps running in the background. You can check it with `ollama list`.

## 2. Install Python packages

You need Python 3.10 or newer.

```bash
cd local-ai-analyst
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Optional: forecasting with Google TimesFM

```bash
pip install -r requirements-forecast.txt
```

This installs PyTorch and TimesFM, about 1 GB. The first forecast downloads the 200M model (about 900 MB) once from Hugging Face. After that, it works offline. Without it, forecast questions still work using a simple built-in baseline, and the chart says so.

## 3. Start the app

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501. The sidebar shows a green message when the model is ready.

## 4. Try it

Click one of the example questions, or type your own. You can add your own files in the sidebar (CSV, Excel, Parquet, JSON). Each file becomes a table named after the file.

## Tips

- **First answer is slow.** The model loads into memory on the first question. After that, each answer takes a few seconds on a 16 GB laptop and longer on CPU-only machines.
- **Clear column names help.** A name like `households_reached` gives much better answers than `col7`.
- **Wrong answer?** Open the **SQL** tab to see what the model did, then rephrase the question more precisely.
- **Use another model:** set `ANALYST_MODEL=llama3.1:8b` before starting, or pick one in the sidebar.
- **Ollama on another machine:** set `OLLAMA_URL=http://that-machine:11434`.
