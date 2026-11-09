$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$app = $null
$doc = $null
try {
    $app = New-Object -ComObject Visio.InvisibleApp
    $app.AlertResponse = 7
    $doc = $app.Documents.Add('')
    $page = $doc.Pages.Item(1)
    $page.PageSheet.CellsU('PageWidth').FormulaU = '16 in'
    $page.PageSheet.CellsU('PageHeight').FormulaU = '11.12 in'
    $shape = $page.DrawRectangle(1,1,2,2)
    $shape.Text = 'Native edit test'
    $shape.CellsU('Char.Font').FormulaU = 'FONT("Times New Roman")'
    [void]$doc.SaveAs((Join-Path $root 'visio_native_reference.vsdx'))
    $doc.Close()
    $doc = $null
    foreach ($name in @('visio_native_reference.vsdx','visio_empty_package_probe.vsdx',
                       'probe_native_ours_document.vsdx','probe_native_ours_pages.vsdx',
                       'probe_native_ours_shapes.vsdx')) {
        try {
            $doc = $app.Documents.Open((Join-Path $root $name))
            [pscustomobject]@{ file=$name; status='OPEN_OK'; shapes=$doc.Pages.Item(1).Shapes.Count } | ConvertTo-Json -Compress
            $doc.Saved=$true
            $doc.Close()
            $doc=$null
        } catch {
            [pscustomobject]@{ file=$name; status='OPEN_FAILED'; error=$_.Exception.Message } | ConvertTo-Json -Compress
        }
    }
} finally {
    if ($null -ne $doc) { $doc.Saved=$true; $doc.Close() }
    if ($null -ne $app) { $app.Quit(); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app) }
}
