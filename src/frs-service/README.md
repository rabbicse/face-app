# FRS Service
## Prerequisites
- Python 3.9
- Pytorch
- Opencv-Python
- Other necessary wheel files

## Installation instructions
- Install Python 3.9, make sure environment variable set properly to windows system
- Open powershell
- Enter the following command:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
- Place your wheels folder in the executing directory
- Then run "install.ps1"
- After finishing all installation keep the models directory to executing directory
- Finally run "run.ps1"

## Notes
```
pip install --no-index --find-links ..\wheels\ -r ..\requirements.txt
```