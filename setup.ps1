winget install UB-Mannheim.TesseractOCR
winget install oschwartz10612.Poppler

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt