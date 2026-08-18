FROM registry.cern.ch/docker.io/cern/alma10-base:latest

WORKDIR /root

ENV TERM=xterm

ENV CC=clang
ENV CXX=clang++

RUN dnf install -y 'dnf-command(config-manager)' \
    && dnf config-manager --set-enabled crb \
    && dnf install -y epel-release \
    && dnf install -y clang gcc g++ eigen3-devel vim wget unzip which htop time libomp-devel \
    && dnf install -y git git-lfs openssl-devel

# install python libraries
RUN dnf install -y python3-pip \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install numpy matplotlib pandas scipy

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v4.1.1/cmake-4.1.1-linux-x86_64.tar.gz \
    && tar -xzvf cmake-4.1.1-linux-x86_64.tar.gz \
    && rm cmake-4.1.1-linux-x86_64.tar.gz
ENV PATH=/root/cmake-4.1.1-linux-x86_64/bin:$PATH

# Install clang-p2996 compiler
# RUN git clone --depth 1 --branch p2996 https://github.com/bloomberg/clang-p2996.git \
#     && cd clang-p2996 \
#     && dnf install -y lld \
#     && mkdir build \
#     && cmake -S llvm -B build -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=/usr/bin/clang -DLLVM_ENABLE_LLD=on \
#         -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=on -DLLVM_APPEND_VC_REV=off -DLLVM_ENABLE_ASSERTIONS=on \
#         -DLLVM_ENABLE_PROJECTS='clang;clang-tools-extra;lld' -DLLVM_ENABLE_RUNTIMES="libc;libcxx;libcxxabi;libunwind"  \
#         -DCMAKE_INSTALL_PREFIX=/data/apps/clang-p2996/install -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_DOCS=OFF      \
#         -DLLVM_BUILD_BENCHMARKS=OFF -DLIBCXX_INSTALL_MODULES=ON -DLLDB_ENABLE_PYTHON=0 \
#     && cmake --build build -j $(nproc) \
#     && cd ..

# Install GCC 16.2.0
RUN mkdir gcc gcc/build gcc/install \
    && cd gcc \
    && git clone --depth 1  --branch releases/gcc-16.2.0 git://gcc.gnu.org/git/gcc.git src \
    && dnf install -y bzip2 flex diffutils \
    && cd src && ./contrib/download_prerequisites \
    && cd ../build \
    && CXX=g++ CC=gcc ../src/configure --prefix=/root/gcc/install --enable-languages=c,c++ --disable-multilib --enable-checking=release \
    && make -j $(nproc) \
    && make install

# install root to run generate_datasets.py
RUN wget https://github.com/root-project/root/releases/download/v6-38-00/root_v6.38.00.Linux-almalinux10.1-x86_64-gcc14.3.tar.gz \
    && dnf install -y xxhash-devel tbb-devel \
    && tar -xvf root_v6.38.00.Linux-almalinux10.1-x86_64-gcc14.3.tar.gz \
    && mv root/ root_v6.38 \
    && rm -rf root_v6.38.00.Linux-almalinux10.1-x86_64-gcc14.3.tar.gz \
    && . root_v6.38/bin/thisroot.sh

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

# Install PAPI
ENV PAPI_VERSION=7-2-0
RUN wget https://github.com/icl-utk-edu/papi/archive/refs/tags/papi-${PAPI_VERSION}-t.tar.gz \
    && mkdir papi \
    && tar -xzf papi-${PAPI_VERSION}-t.tar.gz -C papi --strip-components=1 \
    && rm papi-${PAPI_VERSION}-t.tar.gz \
    && cd papi/src \
    && ./configure \
    && make -j $(nproc) \
    && make install

# Install Google benchmark
ENV GBENCHMARK_VERSION=1.9.5
RUN wget https://github.com/google/benchmark/archive/refs/tags/v${GBENCHMARK_VERSION}.tar.gz \
    && mkdir gbenchmark \
    && tar -xzf v${GBENCHMARK_VERSION}.tar.gz -C gbenchmark --strip-components=1 \
    && rm v${GBENCHMARK_VERSION}.tar.gz \
    && cd gbenchmark \
    && cmake . -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=RelWithDebInfo \
               -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DBENCHMARK_ENABLE_LIBPFM=on \
    && make -j $(nproc)

# Download datasets
RUN wget https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m.zip && unzip 3m.zip && rm -f 3m.zip
RUN wget https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m_v2.zip && unzip 3m_v2.zip && rm -f 3m_v2.zip

# Install the benchmarks repo
ENV DLB_VERSION=556067d
RUN git clone https://github.com/jolly-chen/data-layout-benchmarks.git \
    && cd data-layout-benchmarks \
    && git checkout ${DLB_VERSION} \
    && cmake . -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc \
    && cd .. && mv 3m /root/data-layout-benchmarks/3m && mv 3m_v2 /root/data-layout-benchmarks/3m_v2

ENV PATH="$PATH:/root/cmake-4.1.1-linux-x86_64/bin:/usr/local/likwid/bin"
ENV LD_LIBRARY_PATH="/root/clang-p2996/build/lib/x86_64-unknown-linux-gnu"

CMD ["tail", "-f", "/dev/null"]

