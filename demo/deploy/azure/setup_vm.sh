#!/usr/bin/env bash
# One-shot setup for the PRISM-Bio demo API on an Azure B2s (Ubuntu 22.04/24.04) VM.
#
#   curl -fsSL https://raw.githubusercontent.com/rishimj/prism-plm/main/demo/deploy/azure/setup_vm.sh | \
#     API_DOMAIN=prism-bio.eastus.cloudapp.azure.com bash
#
# Requirements: inbound TCP 80 and 443 open in the VM's network security group,
# and API_DOMAIN resolving to the VM (Azure "DNS name label" works).
set -euo pipefail

API_DOMAIN="${API_DOMAIN:?set API_DOMAIN, e.g. prism-bio.eastus.cloudapp.azure.com}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-https://rishimj.github.io}"
REPO_URL="${REPO_URL:-https://github.com/rishimj/prism-plm.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-/opt/prism-plm}"

echo "==> Installing Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi

echo "==> Adding 2 GB swap (B2s has 4 GB RAM; building the image is memory hungry)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> Fetching code"
if [ -d "$APP_DIR/.git" ]; then
  sudo git -C "$APP_DIR" fetch origin "$BRANCH" && sudo git -C "$APP_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
else
  sudo git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$APP_DIR"
fi

echo "==> Writing configuration"
sudo tee "$APP_DIR/demo/deploy/.env" >/dev/null <<EOF
API_DOMAIN=$API_DOMAIN
ALLOWED_ORIGINS=$ALLOWED_ORIGINS
RATE_LIMIT_PER_MINUTE=30
TORCH_THREADS=2
MAX_SEQUENCE_LENGTH=400
EOF

echo "==> Building and starting containers (first build takes ~5 minutes)"
cd "$APP_DIR/demo/deploy"
sudo docker compose up -d --build

echo "==> Waiting for the API"
for i in $(seq 1 60); do
  if curl -fsS "https://$API_DOMAIN/api/health" >/dev/null 2>&1; then
    echo "API is live at https://$API_DOMAIN/api/health"
    exit 0
  fi
  sleep 5
done
echo "API did not answer over HTTPS yet. Check: sudo docker compose -f $APP_DIR/demo/deploy/docker-compose.yml logs"
exit 1
