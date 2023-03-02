from setuptools import setup, find_packages

setup(
    name='ttsapi',
    version='0.1',
    packages=find_packages(),
    description='TTS service through API',
    install_requires=[
        'numpy',
        'scipy',
        'torchaudio',
        'speechbrain',
        'requests',
        'pyyaml'
    ]
)