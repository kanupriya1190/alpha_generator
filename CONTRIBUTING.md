# Contributing

Thanks for your interest in improving this project.

## Development setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest
cp .env.example .env
```

## Before opening a PR

- Run `pytest`
- Run the dashboard locally with `streamlit run dashboard.py`
- Keep secrets out of commits (`.env` is ignored)

## Pull request guidelines

- Use clear commit messages
- Describe why the change is needed
- Include test coverage for behavior changes
