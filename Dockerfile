FROM python:3.10-slim-bullseye

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "shrimpbot.py"]
