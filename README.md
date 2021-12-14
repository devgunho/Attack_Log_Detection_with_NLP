### How To Run
```
# Create conda env. with specific python version (3.8)
conda create -n "log-nlp" python=3.8

# Check env. list
conda env list

# Activate
conda activate "log-nlp"

# Install requirements.txt
pip install -r requirements.txt
```
```
# Additional
# Make package list text file
pip list --format=freeze > requirements.txt

# Deactivate
conda deactivate
```