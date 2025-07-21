from setuptools import setup, find_packages

setup(
    name="stock-price-predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.1',
        'yfinance==0.2.36',
        'tensorflow-cpu==2.15.0',
        'scikit-learn==1.4.2',
        'matplotlib==3.8.3',
        'streamlit==1.32.2',
        'python-dateutil==2.9.0.post0',
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
