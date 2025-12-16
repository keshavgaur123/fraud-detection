
for linux/mac os
# fraud-detection
#for data/creaditcard.csv //https://www.kaggle.com/datasets/arockiaselciaa/creditcardcsv
first of all set env 
python3 -m venv myenv
source myenv/bin/activate
pip install -r python/requirements.txt

if pack was not install or not found then goto cd python/ run command: for linux python3 model.py 
if done+ deactivate

//------------
windows
cd \path\to\fraud-detection
myenv\Scripts\activate
python -m venv myenv

if done deactivate


Summary

On macOS / Linux, use python3 -m venv myenv and source myenv/bin/activate.

On Windows, use python -m venv myenv and myenv\Scripts\activate (or Activate.ps1 for PowerShell).

Always include myenv/ in .gitignore.

Use pip freeze > requirements.txt to record dependencies.
