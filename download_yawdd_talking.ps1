$ErrorActionPreference = "Stop"

$dataset = "enider/yawdd-dataset"

$root = Join-Path $PSScriptRoot "data\raw\yawdd"
$talkingDir = Join-Path $root "talking"

New-Item -ItemType Directory -Force -Path $talkingDir | Out-Null

$files = @(
    "Mirror/Mirror/Female_mirror/1-FemaleNoGlasses-Talking.avi",
    "Mirror/Mirror/Female_mirror/2-FemaleNoGlasses-Talking.avi",
    "Mirror/Mirror/Female_mirror/6-FemaleNoGlasses-Talking.avi",
    "Mirror/Mirror/Female_mirror/9-FemaleNoGlasses-Talking.avi",
    "Mirror/Mirror/Female_mirror/10-FemaleNoGlasses-Talking.avi",

    "Mirror/Mirror/Male_mirror Avi Videos/4-MaleNoGlasses-Talking.avi",
    "Mirror/Mirror/Male_mirror Avi Videos/5-MaleNoGlasses-Talking.avi",
    "Mirror/Mirror/Male_mirror Avi Videos/6-MaleNoGlasses-Talking.avi",
    "Mirror/Mirror/Male_mirror Avi Videos/9-MaleNoGlasses-Talking.avi",
    "Mirror/Mirror/Male_mirror Avi Videos/15-MaleNoGlasses-Talking.avi"
)

foreach ($item in $files) {
    $leaf = Split-Path $item -Leaf
    $target = Join-Path $talkingDir $leaf

    if (Test-Path $target) {
        Write-Host "[YA EXISTE] $leaf"
        continue
    }

    Write-Host ""
    Write-Host "============================================"
    Write-Host "Descargando: $leaf"
    Write-Host "Clase: talking"
    Write-Host "============================================"

    kaggle datasets download $dataset `
        -f $item `
        -p $root `
        --unzip

    if ($LASTEXITCODE -ne 0) {
        throw "Error descargando $item"
    }

    $relativePath = $item -replace "/", "\"
    $downloaded = Join-Path $root $relativePath

    if (-not (Test-Path $downloaded)) {
        $found = Get-ChildItem `
            -Path $root `
            -Recurse `
            -File `
            -Filter $leaf |
            Select-Object -First 1

        if ($null -eq $found) {
            throw "No se encontro el archivo descargado: $leaf"
        }

        $downloaded = $found.FullName
    }

    Move-Item `
        -Path $downloaded `
        -Destination $target `
        -Force

    Write-Host "[OK] $leaf"
}

$mirrorTemp = Join-Path $root "Mirror"

if (Test-Path $mirrorTemp) {
    Remove-Item $mirrorTemp -Recurse -Force
}

$count = (
    Get-ChildItem $talkingDir -Filter "*.avi" -File
).Count

Write-Host ""
Write-Host "============================================"
Write-Host "DESCARGA TALKING TERMINADA"
Write-Host "============================================"
Write-Host "Talking: $count"
