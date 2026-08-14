<#
    run_real.ps1 — nạp .env rồi chạy vòng luyện tập với model thật.

    Vì sao cần script này: harness KHÔNG tự đọc .env (python-dotenv bị
    comment trong requirements.txt), và arena/model.py:764-767 đọc thẳng
    os.environ với ba tên ARENA_API_KEY / ARENA_BASE_URL / ARENA_MODEL.
    Script này chỉ nạp .env vào biến môi trường của tiến trình hiện tại
    rồi chuyển tiếp mọi tham số cho scripts/run_practice.py. Nó KHÔNG
    sửa file nào trong arena/, data/ hay tests/.

    Dùng:
        .\run_real.ps1                                    # cả 9 brief
        .\run_real.ps1 --brief pub-01-sla-hien-hanh       # một brief
        .\run_real.ps1 --layers none --tag baseline --out runs/real-base.json

    Mặc định thêm --model real --prompt-addendum (đường chạy giống vòng
    tính điểm). Muốn chạy mock thì gọi thẳng python scripts/run_practice.py.
#>

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Khong tim thay .env o $envFile"
}

foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
    $trimmed = $line.Trim()
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
    $parts = $trimmed -split "=", 2
    if ($parts.Count -ne 2) { continue }
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if ($name -ne "") { Set-Item -Path "Env:$name" -Value $value }
}

foreach ($required in @("ARENA_API_KEY", "ARENA_BASE_URL", "ARENA_MODEL")) {
    if (-not (Get-Item -Path "Env:$required" -ErrorAction SilentlyContinue)) {
        throw "Thieu $required trong .env"
    }
}

# stdout tieng Viet tren console Windows
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host ("model     : " + $env:ARENA_MODEL)
Write-Host ("endpoint  : " + $env:ARENA_BASE_URL)
Write-Host  "api key   : da nap tu .env (khong in ra)"

python scripts/run_practice.py --model real --prompt-addendum @args
exit $LASTEXITCODE
