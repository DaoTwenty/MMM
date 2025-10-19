# build_wheels_windows.ps1
param(
    [int[]] $PyVersions = @(8,9,10,11,12),
    [string] $Arch = "AMD64",
    [switch] $Test
)

Write-Host "Building wheels for Windows $Arch..."

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Install requirements
python -m pip install --upgrade cibuildwheel==2.22.0 scikit-build-core pybind11 pytest

$env:CIBW_PLATFORM="windows"
$env:CIBW_ARCHS_WINDOWS=$Arch

if ($Test) {
    $env:CIBW_TEST_REQUIRES="pytest"
    $env:CIBW_TEST_COMMAND="pytest {project}"
}

foreach ($py in $PyVersions) {
    Write-Host "Building Python 3.$py..."
    $env:CIBW_BUILD="cp3$py-*"
    $skip = $PyVersions | Where-Object { $_ -ne $py } | ForEach-Object { " cp3$_-*" }
    $env:CIBW_SKIP=$skip
    python -m cibuildwheel --output-dir "wheelhouse/windows"
}
