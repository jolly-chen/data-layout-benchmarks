FROM almalinux:10
WORKDIR /root

ENV TERM=xterm

ENV CC=clang
ENV CXX=clang++

RUN dnf install -y 'dnf-command(config-manager)' epel-release \
    && dnf config-manager --set-enabled crb

RUN dnf install -y git cmake clang perl wget unzip

# install python libraries
RUN dnf install -y python3-pip \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install numpy matplotlib pandas

# Install screen
RUN dnf install -y screen && chmod 777 /run/screen

# Add Tini
ENV TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Install likwid
ENV LIKWID_VERSION=5.5.1
RUN wget https://github.com/RRZE-HPC/likwid/archive/refs/tags/v${LIKWID_VERSION}.tar.gz \
    && tar -xzf v${LIKWID_VERSION}.tar.gz \
    && rm v${LIKWID_VERSION}.tar.gz \
    && cd likwid-${LIKWID_VERSION} \
    && make -j $(nproc)\
    && make install
ENV PATH=$PATH:"/usr/local/likwid/bin"

# Install papi
ENV PAPI_VERSION=7.2.0
RUN wget https://github.com/icl-utk-edu/papi/releases/download/papi-7-2-0-t/papi-${PAPI_VERSION}.tar.gz \
    && tar -xzf papi-${PAPI_VERSION}.tar.gz \
    && rm papi-${PAPI_VERSION}.tar.gz \
    && cd papi-${PAPI_VERSION}/src \
    && ./configure \
    && make -j $(nproc) \
    && make install

# Copy benchmarks repo
COPY src/ /root/src/

# Download datasets
RUN cd src && mkdir datasets && cd datasets \
    && wget https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m.zip && unzip 3m.zip && rm -f 3m.zip \
    && wget https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m_v2.zip && unzip 3m_v2.zip && rm -f 3m_v2.zip

COPY run_in_container.sh /root/src/run_in_container.sh
RUN chmod +x /root/src/run_in_container.sh

CMD ["tail", "-f", "/dev/null"]