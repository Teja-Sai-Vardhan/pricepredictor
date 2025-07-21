from setuptools import setup, find_packages

setup(
    name="stock-price-predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy<2.0.0',
        'pandas<2.1.0',
        'yfinance>=0.2.0',
        'tensorflow-cpu==2.12.0',
        'scikit-learn<1.4.0',
        'matplotlib<4.0.0',
        'streamlit>=1.10.0',
        'python-dateutil>=2.8.2',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A stock price prediction tool using LSTM neural networks",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/stock-price-predictor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
