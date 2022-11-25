from setuptools import setup

setup(name='mrsearch_IG_RL',
      version='0.1.0',
      install_requires=[
            'gym',
            'numpy',
            'pybullet',
            'datetime',
            'stable_baselines3[extra]',
            'torch',
            'matplotlib',
            'pyyaml',
            'argparse',
            'tqdm',
            'rich'
            ]
)