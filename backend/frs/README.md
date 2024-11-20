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

## Build instructions
https://packaging.python.org/guides/distributing-packages-using-setuptools/#setup-py
```
python dnn_utils_setup.py bdist_wheel --universal
python vision_utils_setup.py bdist_wheel --universal
python retina_face_setup.py bdist_wheel --universal
python arc_faces_setup.py bdist_wheel --universal
```

## Special Notes
```
pip install --no-index --find-links ..\wheels\ -r ..\requirements.txt
```

## For macOS
```
python3 -m pip install --upgrade certifi
open /Applications/Python\ 3.9/Install\ Certificates.command
```