FROM python:3.12.0-slim

# Set the working directory
WORKDIR /app

# Install virtualenv
RUN pip install --no-cache-dir virtualenv

# Create a virtual environment
RUN virtualenv venv

# Activate the virtual environment and install dependencies
COPY requirements.txt .
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
RUN rm config.json
COPY ./constants/constants.ini /app/constants/constants.ini
COPY ./constants/constants.ini /app/constants/constants_template.ini
COPY ./config.json_template /app/config.json 

# Install python-pptx
RUN ./nltk.sh

# Expose the port the app runs on
EXPOSE 8000

# Run the application using the virtual environment
CMD ["sh", "-c", ". venv/bin/activate && python configservice.py"]
