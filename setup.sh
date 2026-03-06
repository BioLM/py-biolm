#!/bin/bash
# Quick setup script for BioLM Pipeline
# This script automates the entire setup process

set -e  # Exit on error

echo "=================================================="
echo "  BioLM Pipeline - Quick Setup Script"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}🐍 Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $PYTHON_VERSION"

if ! python3 -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Python 3.9+ required${NC}"
    exit 1
fi
echo ""

# Install uv if not present
echo -e "${BLUE}📦 Checking for uv...${NC}"
if ! command -v uv &> /dev/null; then
    echo "   Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "   ✅ uv installed"
    
    # Source the shell config to get uv in PATH
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc
    elif [ -f ~/.zshrc ]; then
        source ~/.zshrc
    fi
else
    echo "   ✅ uv is already installed ($(uv --version))"
fi
echo ""

# Check for direnv
echo -e "${BLUE}🔧 Checking for direnv...${NC}"
if command -v direnv &> /dev/null; then
    echo "   ✅ direnv is installed"
    DIRENV_AVAILABLE=true
else
    echo "   ⚠️  direnv not found (optional)"
    echo "   To install: sudo apt install direnv"
    DIRENV_AVAILABLE=false
fi
echo ""

# Create virtual environment
echo -e "${BLUE}🚀 Creating virtual environment...${NC}"
if [ ! -d .venv ]; then
    uv venv --python $(cat .python-version)
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}⚡ Activating environment...${NC}"
source .venv/bin/activate
echo "   ✅ Environment activated"
echo ""

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
echo "   This may take a few minutes..."
uv pip install -e ".[all,dev]"
echo "   ✅ All dependencies installed"
echo ""

# Fix NumPy version if needed
echo -e "${BLUE}🔧 Ensuring NumPy compatibility...${NC}"
uv pip install 'numpy<2.0'
echo "   ✅ NumPy version fixed"
echo ""

# Run tests
echo -e "${BLUE}🧪 Running tests...${NC}"
if python tests/run_tests.py; then
    echo -e "${GREEN}   ✅ All tests passed!${NC}"
else
    echo -e "${YELLOW}   ⚠️  Some tests failed (this is OK if API not configured)${NC}"
fi
echo ""

# Setup direnv if available
if [ "$DIRENV_AVAILABLE" = true ]; then
    echo -e "${BLUE}🔒 Setting up direnv...${NC}"
    direnv allow
    echo "   ✅ direnv configured"
    echo ""
fi

# Final messages
echo "=================================================="
echo -e "${GREEN}  ✅ Setup Complete!${NC}"
echo "=================================================="
echo ""
echo "📚 Next steps:"
echo ""
if [ "$DIRENV_AVAILABLE" = true ]; then
    echo "   1. Exit and re-enter directory (direnv will auto-activate)"
else
    echo "   1. Activate environment: source .venv/bin/activate"
fi
echo "   2. Try an example: make example-simple"
echo "   3. Read the docs: PIPELINE_QUICKSTART.md"
echo "   4. Start coding!"
echo ""
echo "💡 Useful commands:"
echo "   • make help          - Show all commands"
echo "   • make test          - Run tests"
echo "   • make example-simple - Run simple example"
echo "   • make clean         - Clean build artifacts"
echo ""
echo "🎉 Happy pipelining!"
echo ""
