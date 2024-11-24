# CUDAバージョンとUbuntuのバージョンを指定
ARG CUDA_VERSION=12.1.1
ARG CUDA_SHORT_VERSION=12.1
ARG UBUNTU_VERSION=22.04
ARG CUDA_PYTORCH_VERSION=2.1.0

# ベースイメージの設定
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION}

# 使用するシェルをbashに設定
SHELL ["/bin/bash", "-c"]

# ビルド引数の再宣言
ARG CUDA_VERSION
ARG CUDA_SHORT_VERSION
ARG CUDA_PYTORCH_VERSION

# 作業ディレクトリを設定
WORKDIR /workspace

# 必要なシステムパッケージのインストール（dos2unixを含む）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        build-essential \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        python3-dev \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        libunwind-dev \
        dos2unix \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# CUDA関連の環境変数の設定
ENV CUDA_HOME=/usr/local/cuda-${CUDA_SHORT_VERSION}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${CUDA_HOME}/bin:${PATH}

# pip、setuptools、wheelのアップグレード
RUN pip install --upgrade pip setuptools wheel

# 必要な環境変数の設定
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

# CUDAバージョンを指定
ARG cuda_version=${CUDA_SHORT_VERSION}

# PyTorchのインストール（指定したCUDAバージョンに対応）
RUN if [ "$cuda_version" = "12.1" ]; then \
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
            --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$cuda_version" = "12.4" ]; then \
        echo "CUDA 12.4 is not supported for PyTorch 2.1.0"; \
        exit 1; \
    else \
        echo "Invalid CUDA version specified. Please choose either 12.1 or 12.4."; \
        exit 1; \
    fi

# その他のパッケージのインストール
RUN pip install --upgrade transformers && \
    pip install tiktoken && \
    pip install sentencepiece && \
    pip install protobuf && \
    pip install ninja einops triton packaging

# mamba-ssmのインストール
RUN pip install mamba-ssm

# causal-conv1dのインストール
RUN pip install causal-conv1d

# attention-gymのクローンとインストール
RUN git clone https://github.com/pytorch-labs/attention-gym.git && \
    cd attention-gym && \
    pip install . && \
    cd ..

# Flash Attentionのビルド済みホイールのダウンロードとインストール
RUN CUDA_PYTORCH_VERSION_SHORT=${CUDA_PYTORCH_VERSION%.*} && \
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch${CUDA_PYTORCH_VERSION_SHORT}cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install flash_attn-2.7.0.post2+cu12torch${CUDA_PYTORCH_VERSION_SHORT}cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

COPY Hymba-1.5B-Base.py /workspace/
COPY Hymba-1.5B-Instruct.py /workspace/

# デフォルトのコマンドを設定
CMD ["/bin/bash"]