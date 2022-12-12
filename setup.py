from setuptools import setup

setup(name='mrsearch_IG_RL',
      version='0.2.0',
      install_requires=[
            'gym',
            'numpy',
            'scipy',
            'pybullet',
            'datetime',
            'stable_baselines3[extra]',
            'torch',
            'matplotlib',
            'pyyaml',
            'argparse',
            'tqdm',
            'rich',
            'torchviz',
            'jupyter'
            ]
)