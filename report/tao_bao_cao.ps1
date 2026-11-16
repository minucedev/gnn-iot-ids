# Tạo bản báo cáo hoàn chỉnh (Word + PDF) bằng một lệnh, không cần cài thêm gì ngoài uv và Microsoft Word:
#     powershell -ExecutionPolicy Bypass -File report\tao_bao_cao.ps1 [-KhongTaiLieuThamKhao]
# Kết quả: report\BAO_CAO_NOP.docx và report\BAO_CAO_NOP.pdf (nội dung lấy từ report\BAO_CAO.md, đã bỏ mọi trích dẫn file nguồn).
param([switch]$KhongTaiLieuThamKhao)
$ErrorActionPreference = "Stop"
$report = $PSScriptRoot
$extra = @()
if ($KhongTaiLieuThamKhao) { $extra += "--no-references" }
# chạy từ thư mục tạm để uv không dò .venv của repo (có thể bị Windows chặn)
Push-Location $env:TEMP
try {
    uv run --no-project --python 3.11 --with pypandoc_binary --with python-docx python "$report\build_report.py" @extra
    if ($LASTEXITCODE -ne 0) { throw "Dựng DOCX thất bại" }
}
finally { Pop-Location }
& powershell -NoProfile -ExecutionPolicy Bypass -File "$report\export_pdf.ps1"
if ($LASTEXITCODE -ne 0) { throw "Xuất PDF thất bại (cần Microsoft Word)" }
Write-Host "Xong: $report\BAO_CAO_NOP.docx và $report\BAO_CAO_NOP.pdf"
