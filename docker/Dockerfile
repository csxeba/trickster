FROM tensorflow/tensorflow:2.1.0-py3

RUN mkdir /workdir

WORKDIR /workdir

RUN git clone https://github.com/csxeba/trickster.git -b dev

ENV PYTHONPATH=/workdir/trickster

WORKDIR /workdir/trickster.git

CMD ["/bin/bash"]
