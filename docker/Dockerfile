FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG python_version="3.11.5"
SHELL ["bash", "-c"]
ENV HOME /root
RUN apt-get update \
    && apt-get -y upgrade
RUN apt-get -y install build-essential libssl-dev libffi-dev libncurses5-dev zlib1g zlib1g-dev libreadline-dev libbz2-dev libsqlite3-dev liblzma-dev \
    curl git unzip
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PATH:$PYENV_ROOT/bin
ENV PATH $PATH:/root/.pyenv/shims
RUN echo 'eval "$(pyenv init -)"' >> $HOME/.bashrc
RUN . ~/.bashrc

WORKDIR /workdir

RUN pyenv install ${python_version} \
    && pyenv global ${python_version} 
RUN pip install -U torch torchvision torchaudio einops accelerate imageio matplotlib
RUN pip install --upgrade setuptools wheel packaging
ADD entrypoint.sh /entrypoint.sh
ENTRYPOINT [ "bash","/entrypoint.sh" ]