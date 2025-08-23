FROM ollama/ollama:0.11.4-rc0

# 必要なソフトのインストール
RUN apt-get update && apt-get install -y \
    git \
    grep \
    curl \
    xz-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uvのインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Node.jsのインストール
ARG NODE_VERSION=24.6.0
RUN export ARCH=$(uname -m | sed 's/aarch64/arm64/' | sed 's/x86_64/x64/') \
    && curl -fsSLO --compressed "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${ARCH}.tar.xz" \
    && tar -xJf "node-v${NODE_VERSION}-linux-${ARCH}.tar.xz" -C /usr/local --strip-components=1 --no-same-owner \
    && rm "node-v${NODE_VERSION}-linux-${ARCH}.tar.xz" \
    && ln -s /usr/local/bin/node /usr/local/bin/nodejs

# Default command
CMD ["/bin/bash"]
