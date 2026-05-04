sudo apt-get install -y cmake clang-20 unzip

LIKWID_VERSION=5.5.1
wget https://github.com/RRZE-HPC/likwid/archive/refs/tags/v${LIKWID_VERSION}.tar.gz
tar -xzf v${LIKWID_VERSION}.tar.gz
rm v${LIKWID_VERSION}.tar.gz
cd likwid-${LIKWID_VERSION}
make -j $(nproc)
make install
cd ..

PAPI_VERSION=7.2.0
wget https://github.com/icl-utk-edu/papi/releases/download/papi-7-2-0-t/papi-${PAPI_VERSION}.tar.gz
dnf install -y libomp-devel
tar -xzf papi-${PAPI_VERSION}.tar.gz
rm papi-${PAPI_VERSION}.tar.gz
cd papi-${PAPI_VERSION}/src
./configure
make -j $(nproc)
make install
cd ..

wget https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m.zip && unzip 3m.zip && rm -f 3m.zip
wget https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m_v2.zip && unzip 3m_v2.zip && rm -f 3m_v2.zip

