This repository contains code and resources for a project that uses **natural language processing techniques** to analyze text data and provide recommendations. Our goal is to develop a system that can accurately classify and understand text, and use this information to make personalized **recommendations to users**.

## Download Files 
Some of the datasets used in this project are too large to upload to GitHub. They must be downloaded and added in the specified formats within the data/raws folder.
- Reviews.csv $\rightarrow$ https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- glove.6B.50d.txt $\rightarrow$ https://nlp.stanford.edu/projects/glove/

## Installation

Before running the code in this repository, you'll need to install the required packages. We recommend doing this in a virtual environment to avoid conflicts with other packages on your system.

1. Create a virtual environment:
```powershell
virtualenv <nombre entorno>
```
2. Activate the virtual environment:
```powershell
<nombre entorno>\Scripts\activate.bat
```
3. Install the required packages from the `requirements.txt` file:
```powershell
pip install -r requirements.txt
```
If you’re using Jupyter to run the code, you’ll also need to set up a kernel for the virtual environment:

4. Install `ipkernel`:
```powershell
pip install ipykernel
```
5. Deactivate and reactivate the virtual environment:
```powershell
deactivate
<nombre entorno>\Scripts\activate.bat
```
6. Create a ipykernel for the virtual environment:
```powershell
python -m ipykernel install --name=<nombre entorno virtual>
```
7. In case you need to uninstall the kernel:
```powershell
jupyter kernelspec uninstall <name_ entorno virtual>.
```

After completing these steps, you should be able to select the kernel for your virtual environment when running the code in Jupyter.

