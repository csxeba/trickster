FROM tensorflow/tensorflow:2.1.0-py3

ARG UID=1000
ENV UID=$UID

ARG USER=user
ENV USER=$USER

RUN pip3 install \
    matplotlib \
    gym \
    procgen \
    imageio \
    imageio-ffmpeg \
    opencv-python \
    jupyter

RUN useradd -m -s /bin/bash -u $UID $USER

RUN mkdir /workspace && chown $USER /workspace

WORKDIR /workspace

CMD ["/bin/bash"]
