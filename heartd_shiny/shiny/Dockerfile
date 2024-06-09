FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Set PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Copy application and requirements files to the container
COPY taylor_app.py /app/app.py
COPY app_tools.py /app/app_tools.py
COPY requirements.txt /app/requirements.txt

# Copy the models directory
COPY models /models

# Install the required Python packages
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 80

# Run the application
ENTRYPOINT ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "80"]
