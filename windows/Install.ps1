echo "Creating virtual environment...";

python -m venv .\venv

echo "Virtual environment created! activating virtual environment...";

.\venv\Scripts\activate

echo "Installing necessary packages"
pip install wheel

pip install .\dev_wheels\vision_utils-1.0.1-py3-none-any.whl --force-reinstall
pip install .\dev_wheels\dnn_utils-1.0.1-py3-none-any.whl --force-reinstall
pip install .\dev_wheels\retina_face-1.0.1-py3-none-any.whl --force-reinstall
pip install .\dev_wheels\arc_face-1.0.1-py3-none-any.whl --force-reinstall

pip install --no-index --find-links .\wheels\ -r .\frs_service\requirements.txt


Write-Host -NoNewLine 'Press any key to continue...';
$null = $Host.UI.RawUI.ReadKey('IncludeKeyDown');