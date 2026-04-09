# setup.py
from pathlib import Path

from setuptools import find_packages, setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="bosonic-disqco",
    version="0.0.6",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy>=1.26,<2.3.4",
                      "qiskit>=2.2.2",
                      "qiskit-aer>=0.14.0",
                      "qiskit-qasm3-import>=0.6.0",
                      "networkx", 
                      "matplotlib", 
                      "pylatexenc", 
                      "jupyter-tikz", 
                      "ipykernel", 
                      "pytest", 
                      "tqdm"],
    python_requires='>=3.10',
)
