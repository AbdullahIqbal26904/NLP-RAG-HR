#!/bin/bash
# EC2 Instance Setup Script for RAG Talent Matching App
# Run this on a fresh Ubuntu 22.04/24.04 EC2 instance
# Usage: bash setup-ec2.sh

set -e

echo "========================================="
echo "  RAG App - EC2 Setup Script"
echo "========================================="

# Update system
echo "[1/5] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
echo "[2/5] Installing Docker..."
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group
sudo usermod -aG docker $USER
echo "Docker installed. You may need to log out and back in for group changes."

# Install git
echo "[3/5] Installing git..."
sudo apt-get install -y git

# Clone the repo
echo "[4/5] Cloning repository..."
if [ -d "$HOME/rag-assignment" ]; then
    echo "Repository already exists, pulling latest..."
    cd "$HOME/rag-assignment"
    git pull
else
    cd "$HOME"
    read -p "Enter your GitHub repo URL: " REPO_URL
    git clone "$REPO_URL" rag-assignment
    cd "$HOME/rag-assignment"
fi

# Setup .env
echo "[5/5] Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "========================================="
    echo "  IMPORTANT: Edit your .env file!"
    echo "========================================="
    echo "Run: nano ~/rag-assignment/.env"
    echo "Fill in your API keys for:"
    echo "  - PINECONE_API_KEY"
    echo "  - HF_API_TOKEN"
    echo "  - GROQ_API_KEY"
    echo ""
else
    echo ".env already exists, skipping."
fi

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Log out and back in (for Docker permissions)"
echo "  2. Edit .env:  nano ~/rag-assignment/.env"
echo "  3. Build:      cd ~/rag-assignment && docker compose -f docker-compose.prod.yml build"
echo "  4. Run:        docker compose -f docker-compose.prod.yml up -d"
echo "  5. Check:      docker compose -f docker-compose.prod.yml logs -f"
echo ""
echo "App will be available at: http://<your-ec2-public-ip>"
echo ""
