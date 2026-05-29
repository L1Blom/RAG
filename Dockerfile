FROM python:3.12.0-slim

# Set the working directory
WORKDIR /app

# Install dependencies directly into the system Python.
# Virtualenv is unnecessary inside a container — the container itself provides isolation.
# setuptools must come first as apscheduler 3.x imports pkg_resources from it.
COPY requirements.txt .
RUN pip install --no-cache-dir setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
COPY ./constants/constants.ini /app/constants/constants.ini
COPY ./constants/constants.ini /app/constants/constants_template.ini
COPY ./config.json_template /app/config.json 

# Download NLTK corpora
RUN ./nltk.sh

# Expose the port the app runs on
EXPOSE 8000

CMD ["python", "configservice.py"]
