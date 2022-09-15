FROM tensorflow/tensorflow:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y init && apt-get clean all
RUN apt update -y
RUN apt dist-upgrade -y
RUN apt autoremove -y
RUN apt-get -y update
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    patch \
    file \
    git \
    curl \
    nkf \
    locales \
    wget \
    emacs \
    unzip \
    mlocate \
    time

RUN apt-get -y update
RUN apt-get update
RUN apt-get install -y mysql-client

RUN apt-get install --reinstall -y libnss3
RUN apt-get install -y language-pack-ja-base language-pack-ja
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV PYTHONIOENCODING utf-8
RUN echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc
RUN echo "export LC_ALL=ja_JP.UTF-8" >> ~/.bashrc
RUN echo "export PYTHONIOENCODING=utf-8" >> ~/.bashrc

RUN apt-get -y update
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

# Tensor Flow and TF Agent
RUN pip3 install tensorflow tf-agents

# 作業領域の作成 (ホストOS の /data をマウントするため)
WORKDIR /data
WORKDIR /opt
COPY program /opt/program/
