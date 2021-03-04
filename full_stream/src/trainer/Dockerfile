FROM continuumio/miniconda3
ADD environment.yml environment.yml
# copy the dependencies file to the working directory
COPY requirements.txt .
RUN conda env create -n rapids -f ./environment.yml

# copy the content of the local src directory to the working directory
COPY . .

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]

# command to run on container start
ENTRYPOINT ["conda", "run", "-n", "rapids", "python3", "main.py"]
