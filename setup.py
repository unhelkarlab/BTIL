from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(name="BTIL",
      version="2.0",
      author="Sangwon Seo",
      author_email="sangwon.seo@rice.edu",
      description="Bayesian Team Imitation Learner",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(exclude=["data", "log"]),
      python_requires='>=3.8',
      install_requires=['numpy', 'tqdm', 'scipy'])
