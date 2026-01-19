#!/usr/bin/env bash
set -e

echo "=== Installing Python 2.7.18 from source ==="

# 1. Download Python 2.7.18
if [ ! -f Python-2.7.18.tgz ]; then
  wget https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz
fi

# 2. Extract
if [ ! -d Python-2.7.18 ]; then
  tar -xvf Python-2.7.18.tgz
fi

cd Python-2.7.18

# 3. Configure & build
./configure --enable-optimizations
make

# 4. Install (SYSTEM-WIDE)
echo ">>> sudo make install (you may be asked for password)"
sudo make install

cd ..

# 5. Verify python2
echo "=== Verifying Python 2 ==="
python2 --version

# 6. Install pip for Python 2
echo "=== Installing pip for Python 2 ==="
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
python2 get-pip.py

# 7. Verify pip
python2 -m pip --version

# 8. Install BNFinder dependencies
echo "=== Installing BNFinder dependencies ==="
python2 -m pip install "numpy==1.16.6"
python2 -m pip install fpconst
python2 -m pip install "scipy==1.2.3"

# 9. Install BNFinder
echo "=== Installing BNFinder ==="
python2 -m pip install BNfinder

# 10. Verify bnf
echo "=== Verifying bnf ==="
which bnf || {
  echo "bnf not found in PATH"
  echo "You may need: export PATH=\"\$HOME/.local/bin:\$PATH\""
  exit 1
}

echo "=== DONE: Python 2.7 + BNFinder installed successfully ==="
