echo "Creating virtual environment...";

python -m venv .\venv

echo "Virtual environment created! activating virtual environment...";

.\venv\Scripts\activate

echo "Changing app directory"
cd .\frs_service
python app.py


Write-Host -NoNewLine 'Press any key to continue...';
$null = $Host.UI.RawUI.ReadKey('IncludeKeyDown');