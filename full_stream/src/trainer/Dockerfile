FROM nvidia/cuda:11.0-runtime-ubuntu20.04

ENV CONDA_PATH=/opt/anaconda3
ENV ENVIRONMENT_NAME=rapids
SHELL ["/bin/bash", "-c"]

# curl is required to download Anaconda.
RUN apt-get update && apt-get install curl -y

# Download and install Anaconda.
RUN cd /tmp && curl -O https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
RUN chmod +x /tmp/Anaconda3-2019.07-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash -c "/tmp/Anaconda3-2019.07-Linux-x86_64.sh -b -p ${CONDA_PATH}"

# Initializes Conda for bash shell interaction.
RUN ${CONDA_PATH}/bin/conda init bash

# Upgrade Conda to the latest version
RUN ${CONDA_PATH}/bin/conda update -n base -c defaults conda -y

# Create the work environment and setup its activation on start.
RUN ${CONDA_PATH}/bin/conda create --name ${ENVIRONMENT_NAME} -y
RUN echo conda activate ${ENVIRONMENT_NAME} >> /root/.bashrc

COPY ./environment.yml /tmp/
COPY ./requirements.txt /tmp/

RUN . ${CONDA_PATH}/bin/activate ${ENVIRONMENT_NAME} \
  && conda env update --file /tmp/environment.yml --prune

# command to run on container start
ENTRYPOINT ["conda", "run", "-n", "rapids", "python3", "main.py"]
