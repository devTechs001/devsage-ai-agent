from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read dev requirements
with open('requirements-dev.txt') as f:
    dev_requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="devsage",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
    },
    entry_points={
        'console_scripts': [
            'devsage=api.main:main',
            'devsage-index=scripts.index_codebase:cli',
        ],
    },
    python_requires='>=3.8',
)
