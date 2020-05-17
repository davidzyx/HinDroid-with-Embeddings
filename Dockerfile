ARG BASE_CONTAINER=jupyter/scipy-notebook:latest
ARG DATAHUB_CONTAINER=ucsdets/datahub-base-notebook:2019.4.9

FROM $DATAHUB_CONTAINER as datahub

FROM $BASE_CONTAINER

USER root

COPY --from=datahub /usr/share/datahub/scripts/* /usr/share/datahub/scripts/
RUN /usr/share/datahub/scripts/install-all.sh

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y default-jre && \
    apt-get install -y default-jdk && \
	apt-get install -y htop

ENV APK_SCRIPT https://raw.githubusercontent.com/iBotPeaches/Apktool/master/scripts/linux/apktool
ENV APK_JAR https://bitbucket.org/iBotPeaches/apktool/downloads/apktool_2.4.1.jar

RUN mkdir -p /usr/local/bin

RUN P=/tmp/$(basename $APK_SCRIPT) && \
    wget -q -O $P $APK_SCRIPT && \
    chmod +x $P && \
    mv $P /usr/local/bin

RUN P=/tmp/$(basename $APK_JAR) && \
    wget -q -O $P $APK_JAR && \
    chmod +x $P && \
    mv $P /usr/local/bin/apktool.jar
    
RUN conda install pytorch torchvision cpuonly -c pytorch

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt  && \
	fix-permissions $CONDA_DIR

COPY --from=datahub /run_jupyter.sh /
