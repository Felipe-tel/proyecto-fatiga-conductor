$ErrorActionPreference = "Stop"

$dataset = "enider/yawdd-dataset"
$root = Join-Path $PSScriptRoot "data\raw\yawdd"

$classes = @("normal", "talking", "yawning")

foreach ($class in $classes) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root $class) | Out-Null
}

$files = @(
    @{ Class = "normal";  Path = "Mirror/Mirror/Female_mirror/11-FemaleNoGlasses-Normal.avi" },
    @{ Class = "talking"; Path = "Mirror/Mirror/Female_mirror/11-FemaleNoGlasses-Talking.avi" },
    @{ Class = "yawning"; Path = "Mirror/Mirror/Female_mirror/11-FemaleNoGlasses-Yawning.avi" },

    @{ Class = "normal";  Path = "Mirror/Mirror/Female_mirror/12-FemaleNoGlasses-Normal.avi" },
    @{ Class = "talking"; Path = "Mirror/Mirror/Female_mirror/12-FemaleNoGlasses-Talking.avi" },
    @{ Class = "yawning"; Path = "Mirror/Mirror/Female_mirror/12-FemaleNoGlasses-Yawning.avi" },

    @{ Class = "normal";  Path = "Mirror/Mirror/Male_mirror Avi Videos/17-MaleNoGlasses-Normal.avi" },
    @{ Class = "talking"; Path = "Mirror/Mirror/Male_mirror Avi Videos/17-MaleNoGlasses-Talking.avi" },
    @{ Class = "yawning"; Path = "Mirror/Mirror/Male_mirror Avi Videos/17-MaleNoGlasses-Yawning.avi" },

    @{ Class = "normal";  Path = "Mirror/Mirror/Male_mirror Avi Videos/18-MaleNoGlasses-Normal.avi" },
    @{ Class = "talking"; Path = "Mirror/Mirror/Male_mirror Avi Videos/18-MaleNoGlasses-Talking.avi" },
    @{ Class = "yawning"; Path = "Mirror/Mirror/Male_mirror Avi Videos/18-MaleNoGlasses-Yawning.avi" }
)

foreach ($item in $files) {
    $leaf = Split-Path $item.Path -Leaf
    $targetDir = Join-Path $root $item.Class
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

Write-Host ""
Write-Host "============================================"
Write-Host "TEST FINAL YAWDD DESCARGADO"
Write-Host "============================================"
Write-Host "Sujetos nuevos: F11, F12, M17, M18"
Write-Host "Videos nuevos: 12"
