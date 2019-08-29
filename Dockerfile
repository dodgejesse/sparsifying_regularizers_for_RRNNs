FROM nvidia/cuda:9.0-cudnn7-devel

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /usr/local/cuda/bin/:/usr/local/nvidia/bin:$PATH

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/nvidia/lib64/

WORKDIR /stage

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN add-apt-repository ppa:jonathonf/python-3.6

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    curl \
    python3.6 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN pip install --upgrade pip

RUN pip install setuptools wheel

RUN pip install --no-cache-dir cupy-cuda90==4.1.0

RUN pip install --no-cache-dir -q https://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip freeze

RUN git clone https://github.com/dodgejesse/rational-recurrences

WORKDIR rational-recurrences/

COPY classification/train_classifier.py classification/train_classifier.py
COPY classification/run_local_experiment.py classification/run_local_experiment.py
COPY language_model/train_lm.py language_model/train_lm.py
COPY classification/save_learned_structure.py classification/save_learned_structure.py
COPY classification/run_beaker_classification.sh classification/run_beaker_classification.sh
COPY classification/run_beaker_structure_learning_classification.sh classification/run_beaker_structure_learning_classification.sh
COPY language_model/run_beaker_lm.sh language_model/run_beaker_lm.sh
COPY rrnn.py rrnn.py
COPY semiring.py semiring.py

CMD ["python3.6", "-u", "classification/run_local_experiment.py"]