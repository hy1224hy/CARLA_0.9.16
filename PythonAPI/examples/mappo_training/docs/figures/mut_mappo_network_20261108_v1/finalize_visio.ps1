param([switch]$ReplaceGenerated)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$draft = Join-Path $root 'mut_mappo_network_editable_draft.vsdx'
$final = Join-Path $root 'mut_mappo_network_editable.vsdx'
$preview = Join-Path $root 'mut_mappo_network_visio_preview.png'
$report = Join-Path $root 'visio_validation.json'
if ((Test-Path -LiteralPath $final) -and -not $ReplaceGenerated) {
    throw 'The output already exists. Use a new name or explicitly replace this generated artifact.'
}
$bindings = Get-Content -LiteralPath (Join-Path $root 'visio_bindings.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$app = $null
$document = $null
try {
    # A new invisible instance leaves any existing user Visio session alone.
    $app = New-Object -ComObject Visio.InvisibleApp
    $app.AlertResponse = 7
    $app.EventsEnabled = $false
    $app.ScreenUpdating = $false
    $document = $app.Documents.Open($draft)
    $page = $document.Pages.Item(1)
    $glued = 0
    foreach ($binding in $bindings) {
        $line = $page.Shapes.ItemFromID([int]$binding.line)
        $target = $page.Shapes.ItemFromID([int]$binding.target)
        $line.CellsU([string]$binding.cell).GlueToPos($target, [double]$binding.x, [double]$binding.y)
        # Custom polylines also need the transform dependency used by native
        # Visio connectors: PAR(PNT(...)) alone tracks connection coordinates,
        # but not the target group's page transform when it is moved.
        $trigger = if ($binding.cell -eq 'BeginX') { 'BegTrigger' } else { 'EndTrigger' }
        $line.CellsU($trigger).FormulaU = ('_XFTRIGGER(Sheet.{0}!EventXFMod)' -f [int]$binding.target)
        $glued++
    }
    [void]$document.SaveAs($final)
    $document.Close()
    $document = $app.Documents.Open($final)
    $page = $document.Pages.Item(1)
    $counts = @{ shapes=0; groups=0; text=0; one_dimensional=0; foreign=0 }
    function Measure-Shapes($collection) {
        foreach ($shape in $collection) {
            $counts.shapes++
            if ($shape.Type -eq 2) { $counts.groups++ }
            if ($shape.Type -eq 4) { $counts.foreign++ }
            if ($shape.OneD -ne 0) { $counts.one_dimensional++ }
            if (-not [string]::IsNullOrWhiteSpace([string]$shape.Text)) { $counts.text++ }
            if ($shape.Shapes.Count -gt 0) { Measure-Shapes $shape.Shapes }
        }
    }
    Measure-Shapes $page.Shapes
    if ($counts.shapes -lt 400 -or $counts.text -lt 60 -or $counts.foreign -gt 0) {
        throw ('Native shape verification failed: ' + ($counts | ConvertTo-Json -Compress))
    }
    $app.Settings.SetRasterExportResolution(3, 180.0, 180.0, 0)
    $page.Export($preview)
    # The separate diagnose_glue.ps1 control found that InvisibleApp returns
    # unchanged cached endpoint coordinates for both a native DrawLine control
    # and this drawing during numeric PinX edits. That headless probe cannot
    # establish interactive drag behavior. Validate the saved native glue
    # records instead; do not claim that GUI dragging was tested.
    $glueVerified = $page.Connects.Count -ge $bindings.Count
    if (-not $glueVerified) { throw 'Saved native connection records are incomplete.' }
    $result = [ordered]@{
        status = 'VISIO_OPEN_SAVE_REOPEN_VERIFIED'
        output = $final
        preview = $preview
        pages = $document.Pages.Count
        native_shapes = $counts.shapes
        groups = $counts.groups
        editable_text_shapes = $counts.text
        editable_lines = $counts.one_dimensional
        foreign_objects = $counts.foreign
        glued_endpoints = $glued
        connector_glue_verified = $glueVerified
        interactive_drag_test = 'Not verified; headless numeric probe also failed on native API control'
        visio_version = $app.Version
    }
    $result | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $report -Encoding UTF8
    $result | ConvertTo-Json -Depth 3
    $document.Saved = $true
    $document.Close()
    $document = $null
}
finally {
    if ($null -ne $document) {
        $document.Saved = $true
        $document.Close()
    }
    if ($null -ne $app) {
        $app.Quit()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)
    }
}
