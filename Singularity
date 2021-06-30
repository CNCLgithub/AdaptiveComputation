bootstrap: docker
from: julia:1.6

%environment
 export PATH=$PATH:/miniconda/bin

%runscript
  exec bash "$@"

%post
 apt-get update
 apt-get install -y software-properties-common
 apt-get install -y  build-essential \
                     graphviz \
                     git \
                     wget \
                     ffmpeg \
                     libglu1 \
                     libxi6 \
                     libc6 \
                     libsm6 \
                     libxrender-dev \
                     libgl1-mesa-dev \
                     gettext \
                     gettext-base \
                     libgtk-3-dev \
                     libglib2.0-dev \
                     xdg-utils \
                     cmake
 apt-get clean

 # Setup conda
 wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O conda.sh
 bash conda.sh -b -p /miniconda
 rm conda.sh

 # Add an sbatch workaround
 echo '#!/bin/bash\nssh -y "$HOSTNAME"  sbatch "$@"'  > /usr/bin/sbatch
 chmod +x /usr/bin/sbatch

 # Add an scancel workaround
 echo '#!/bin/bash\nssh -y "$HOSTNAME"  scancel "$@"'  > /usr/bin/scancel
 chmod +x /usr/bin/scancel

 # Add an srun workaround
 echo '#!/bin/bash\nssh -y "$HOSTNAME"  srun "$@"'  > /usr/bin/srun
 chmod +x /usr/bin/srun
