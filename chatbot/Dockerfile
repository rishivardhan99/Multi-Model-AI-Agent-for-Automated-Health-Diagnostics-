FROM python:3.10-slim

# set working directory INSIDE chatbot
WORKDIR /app/chatbot

# install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy ONLY chatbot folder contents
COPY . .

# install chatbot deps
RUN pip install --no-cache-dir -r requirements.txt

# expose streamlit port
EXPOSE 8501

# IMPORTANT: run streamlit from THIS folder
CMD ["streamlit", "run", "dev_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
