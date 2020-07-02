FROM tensorflow/tensorflow:1.8.0-gpu-py3
#FROM tensorflow/tensorflow:latest-gpu-py3

ENV SRC_DIR /src
ENV PORT 8888

RUN mkdir -p $SRC_DIR
COPY requirements.txt $SRC_DIR
WORKDIR $SRC_DIR
#VOLUME ["/data", "/logs"]
SHELL ["/bin/bash", "-c"]

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y libc6 libsm6 libxext6 libxrender-dev libglib2.0-0 && pip install opencv-python
RUN pip install -r requirements.txt

#RUN python -m ipykernel install --user --display-name gaeun_python3

RUN pip install jupyter
RUN jupyter notebook --generate-config --allow-root -y
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = $PORT" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py

EXPOSE $PORT
#ENTRYPOINT jupyter notebook --ip=0.0.0.0 --port=$PORT --no-browser --allow-root
#ENTRYPOINT jupyter notebook --allow-root