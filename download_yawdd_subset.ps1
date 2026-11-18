$ErrorActionPreference = "Stop"

$dataset = "enider/yawdd-dataset"

$root = Join-Path $PSScriptRoot "data\raw\yawdd"
$normalDir = Join-Path $root "normal"
$yawningDir = Join-Path $root "yawning"

New-Item -ItemType Directory -Force -Path $normalDir | Out-Null
New-Item -ItemType Directory -Force -Path $yawningDir | Out-Null

$files = @(
    # =========================
    # MUJERES
    # =========================

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Female_mirror/1-FemaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Female_mirror/1-FemaleNoGlasses-Yawning.avi"
    },

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Female_mirror/2-FemaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Female_mirror/2-FemaleNoGlasses-Yawning.avi"
    },

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Female_mirror/6-FemaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Female_mirror/6-FemaleNoGlasses-Yawning.avi"
    },

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Female_mirror/9-FemaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Female_mirror/9-FemaleNoGlasses-Yawning.avi"
    },

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Female_mirror/10-FemaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Female_mirror/10-FemaleNoGlasses-Yawning.avi"
    },

    # =========================
    # HOMBRES
    # =========================

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/4-MaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/4-MaleNoGlasses-Yawning.avi"
    },

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/5-MaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/5-MaleNoGlasses-Yawning.avi"
    },

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/6-MaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/6-MaleNoGlasses-Yawning.avi"
    },

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/9-MaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/9-MaleNoGlasses-Yawning.avi"
    },

    @{
        Class = "normal"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/15-MaleNoGlasses-Normal.avi"
    },
    @{
        Class = "yawning"
        Path = "Mirror/Mirror/Male_mirror Avi Videos/15-MaleNoGlasses-Yawning.avi"
    }
)

foreach ($item in $files) {

    $leaf = Split-Path $item.Path -Leaf

    if ($item.Class -eq "normal") {
        $targetDir = $normalDir
    }
    else {
        $targetDir = $yawningDir
    }

    $target = Join-Path $targetDir $leaf

    if (Test-Path $target) {
        Write-Host "[YA EXISTE] $leaf"
        continue
    }

    Write-Host ""
    Write-Host "============================================"
    Write-Host "Descargando: $leaf"
    Write-Host "Clase: $($item.Class)"
    Write-Host "============================================"

    kaggle datasets download $dataset `
        -f $item.Path `
        -p $root `
        --unzip

    if ($LASTEXITCODE -ne 0) {
        throw "Error descargando $($item.Path)"
    }

    $relativePath = $item.Path -replace "/", "\"
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

$normalCount = (
    Get-ChildItem $normalDir -Filter "*.avi" -File
).Count

$yawningCount = (
    Get-ChildItem $yawningDir -Filter "*.avi" -File
).Count

Write-Host ""
Write-Host "============================================"
Write-Host "DESCARGA YAWDD TERMINADA"
Write-Host "============================================"
Write-Host "Normal:  $normalCount"
Write-Host "Yawning: $yawningCount"
Write-Host "Total:   $($normalCount + $yawningCount)"