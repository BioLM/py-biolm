pip install -r requirements_dev.txt
make dist
twine check dist/*
make install

