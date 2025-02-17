# Use Python 3.10.14 as the base image
FROM python:3.10.14
 
# Create and set /app as the working directory
WORKDIR /app
 
# Copy requirements.txt into the container
COPY requirements.txt /app/
 
# Install dependencies (no cache to keep the image size small)
RUN pip install -r /app/requirements.txt
 
# Copy the rest of the project files into /app
# This will include your my_project/ folder, tests/, etc.
COPY . /app/
 
# Set the default command to run the pipeline via main.py
ENTRYPOINT ["python", "-m", "histo-infer.main"]

# FROM python:3.10.14

# WORKDIR /app

# # Install pip requirements
# COPY requirements.txt /app/requirements.txt
# RUN pip install -r /app/requirements.txt



# COPY pipeline_infer.py /app/pipeline_infer.py

# ENTRYPOINT ["python", "pipeline_infer.py"]