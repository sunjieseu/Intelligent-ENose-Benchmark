from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="intelligent-enose-benchmark",
    version="1.0.0",
    author="Jie Sun, Wenjie Luo, Zhongshan Chen",
    author_email="jie.sun@njxzc.edu.cn",
    description="A comprehensive benchmark for robust intelligent E-nose systems under data scarcity and sensor drift",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunjieseu/Intelligent-ENose-Benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords=[
        "electronic nose",
        "transfer learning",
        "few-shot learning",
        "drift compensation",
        "gas sensing",
        "machine learning",
        "benchmark",
    ],
)
