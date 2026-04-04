from setuptools import setup, find_packages

setup(
    name="hcderiv",
    version="0.3.0",
    author="Zetta Byte",
    description="Exact derivatives via hypercomplex perturbation algebra",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.20", "scipy>=1.7"],
    extras_require={
        "viz": ["matplotlib>=3.4"],
        "jax": ["jax>=0.4", "jaxlib>=0.4"],
        "dev": ["pytest", "sphinx"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
