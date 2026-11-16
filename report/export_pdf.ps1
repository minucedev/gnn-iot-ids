# Mở BAO_CAO_NOP.docx bằng Microsoft Word, cập nhật mục lục và số trang, lưu lại docx, xuất BAO_CAO_NOP.pdf.
# Chạy:  powershell -ExecutionPolicy Bypass -File report\export_pdf.ps1   (cần Microsoft Word đã cài; không cần mở sẵn)
param(
    [string]$Docx = (Join-Path $PSScriptRoot "BAO_CAO_NOP.docx"),
    [string]$Pdf = (Join-Path $PSScriptRoot "BAO_CAO_NOP.pdf")
)
$Docx = (Resolve-Path $Docx).Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $false)
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }      # lần hai: số trang sau khi mục lục đổi độ dài
    $doc.Repaginate()
    $doc.Save()
    # 17 = wdExportFormatPDF; 0 = tối ưu cho in; 1 = chỉ tài liệu; tạo dấu trang từ các tiêu đề (heading)
    $doc.ExportAsFixedFormat($Pdf, 17, $false, 0, 0, 1, 1, 0, $true, 1, 0, $true, $true, $false)
    $pages = $doc.ComputeStatistics(2)
    $doc.Close($false)
    Write-Host "Đã ghi $Pdf ($pages trang)"
}
finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
