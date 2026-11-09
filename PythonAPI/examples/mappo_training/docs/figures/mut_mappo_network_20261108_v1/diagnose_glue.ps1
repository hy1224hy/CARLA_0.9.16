$ErrorActionPreference='Stop'
$app=$null
$doc=$null
try {
    $app=New-Object -ComObject Visio.InvisibleApp
    $app.AlertResponse=7
    $app.EventsEnabled=$true
    function Check-Move($label,$line,$target,$cell) {
        $pin=[double]$target.CellsU('PinX').ResultIU
        $old=[double]$line.CellsU($cell).ResultIU
        $target.CellsU('PinX').ResultIU=$pin+0.125
        $current=[double]$line.CellsU($cell).ResultIU
        [pscustomobject]@{kind=$label;screen_updating=$app.ScreenUpdating;events=$app.EventsEnabled;
            one_d=$line.OneD;old=$old;new=$current;delta=($current-$old);
            begin_trigger=$line.CellsU('BegTrigger').FormulaU} | ConvertTo-Json -Compress
        $target.CellsU('PinX').ResultIU=$pin
    }
    $doc=$app.Documents.Add('')
    $page=$doc.Pages.Item(1)
    $rect=$page.DrawRectangle(2,2,4,4)
    $line=$page.DrawLine(3,2,5,1)
    $line.CellsU('BeginX').GlueToPos($rect,0.5,0)
    foreach ($flag in @($false,$true)) {
        $app.ScreenUpdating=$flag
        Check-Move 'native_api_control' $line $rect 'BeginX'
    }
    $doc.Saved=$true
    $doc.Close()
    $doc=$app.Documents.Open((Join-Path $PSScriptRoot 'mut_mappo_network_editable.vsdx'))
    $page=$doc.Pages.Item(1)
    $bindings=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'visio_bindings.json') -Raw | ConvertFrom-Json
    $b=$bindings[0]
    $line=$page.Shapes.ItemFromID([int]$b.line)
    $target=$page.Shapes.ItemFromID([int]$b.target)
    foreach ($flag in @($false,$true)) {
        $app.ScreenUpdating=$flag
        Check-Move 'generated_group' $line $target ([string]$b.cell)
    }
    $doc.Saved=$true
    $doc.Close()
    $doc=$null
} finally {
    if ($null -ne $doc) { $doc.Saved=$true; $doc.Close() }
    if ($null -ne $app) { $app.Quit(); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app) }
}
