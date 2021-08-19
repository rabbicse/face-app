$location = Get-Location

echo "Hello";

python -m venv .\venv

.\venv\Scripts\activate

pip install wheel

echo "venv created!";


Write-Host -NoNewLine 'Press any key to continue...';
$null = $Host.UI.RawUI.ReadKey('IncludeKeyDown');