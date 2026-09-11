[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputDirectory = (Join-Path (Get-Location) "migration-output"),
    [long]$PartBytes = 536870912,
    [string]$ExpectedSha256 = "cb9ef494946f4ca43a883768904df234fd2f57b36e46c3cdc61f29c0957337f2",
    [long]$ExpectedBytes = 14067722240
)

$ErrorActionPreference = "Stop"
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$repoPath = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$archiveInfo = Get-Item -LiteralPath $archivePath
if ($archiveInfo.Length -ne $ExpectedBytes) { throw "Archive size mismatch: $($archiveInfo.Length)" }
$archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
if ($archiveHash -ne $ExpectedSha256.ToLowerInvariant()) { throw "Archive SHA-256 mismatch" }

$entries = @(& tar -tf $archivePath)
if ($LASTEXITCODE -ne 0 -or $entries.Count -eq 0) { throw "Unable to enumerate archive" }
foreach ($entry in $entries) {
    $normalized = $entry.Replace('\', '/')
    if ($normalized.StartsWith('/') -or $normalized -match '^[A-Za-z]:' -or $normalized.Split('/') -contains '..') {
        throw "Unsafe archive entry: $entry"
    }
}

function Test-Allowlisted([string]$Name) {
    $n = $Name.Replace('\', '/')
    if ($n -match '(^|/)(wandb|cache|__pycache__|\.git|\.pytest_cache)(/|$)') { return $false }
    if ($n -match '(^|/)(\.env($|\.)|id_rsa|id_ed25519|wireguard|wg0|api[_-]?key)') { return $false }
    if ($n -match '^sac-qutab/(src|tests|docs|scripts|configs)/') { return $true }
    if ($n -match '^sac-qutab/(README\.md|pyproject\.toml|requirements\.txt|final_project_reference\.md|project_validation\.md|qutab_reference\.md|SAC-QUTAB_Detailed_Proposal\.pdf|dl_final_project\.pdf)$') { return $true }
    if ($n -match '^data/eurosat/') { return $true }
    if ($n -match '^weights/(resnet50\.tv_in1k|vit_small_patch16_224\.augreg_in1k)\.safetensors$') { return $true }
    if ($n -match '^artifacts/data/') { return $true }
    if ($n -match '^artifacts/counterfactuals/(validation|calibration|final_test)(\.summary)?\.jsonl?$') { return $true }
    if ($n -match '^artifacts/review/[^/]+$') { return $true }
    if ($n -match '^artifacts/runs/resnet50-seed(17|29|43)/(resolved_run\.json|environment\.json|model\.json|events\.jsonl|summary\.json|wandb_run\.json|checkpoints/best\.pt)$') { return $true }
    if ($n -match '^artifacts/evaluations/resnet50-seed(17|29|43)/(validation|calibration)/') { return $true }
    if ($n -match '^artifacts/detectors/(resnet50-seed(17|29|43)\.json|wandb_resnet50-seed(17|29|43)_detector-fit\.json)$') { return $true }
    if ($n -match '^artifacts/preflight/(nvidia-smi\.txt|environment\.txt|pip-freeze\.txt|config\.json|hardware\.json|wandb_five-model_hardware-preflight\.json|resnet50_weights\.json|vit_small_patch16_224_weights\.json)$') { return $true }
    if ($n -match '^artifacts/preflight/resnet50/(summary\.json|resolved_run\.json|environment\.json|model\.json|events\.jsonl|wandb_run\.json)$') { return $true }
    return $false
}

# Never pass directory members to tar: selecting a directory makes bsdtar recurse
# into excluded children (notably W&B symlinks/logs) despite the file allowlist.
$selected = @($entries | Where-Object {
    $normalized = $_.Replace('\', '/')
    -not $normalized.EndsWith('/') -and (Test-Allowlisted $normalized)
})
if ($selected.Count -lt 27000) { throw "Allowlist unexpectedly selected only $($selected.Count) entries" }
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $outputPath) {
    if ((Get-ChildItem -LiteralPath $outputPath -Force | Measure-Object).Count -gt 0) { throw "Output directory must be empty: $outputPath" }
} else {
    New-Item -ItemType Directory -Path $outputPath | Out-Null
}
$staging = Join-Path $outputPath "staging"
New-Item -ItemType Directory -Path $staging | Out-Null
$listPath = Join-Path $outputPath "archive-allowlist.txt"
$selected | Set-Content -LiteralPath $listPath -Encoding utf8
& tar -xf $archivePath -C $staging -T $listPath
if ($LASTEXITCODE -ne 0) { throw "Allowlisted extraction failed" }

# Overlay the current implementation so the A100 receives the campaign-aware code, not stale archived code.
$repoDestination = Join-Path $staging "sac-qutab"
foreach ($name in @("src", "tests", "docs", "scripts", "configs")) {
    Copy-Item -LiteralPath (Join-Path $repoPath $name) -Destination $repoDestination -Recurse -Force
}
foreach ($name in @("README.md", "pyproject.toml", "requirements.txt")) {
    $source = Join-Path $repoPath $name
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $repoDestination -Force }
}
Get-ChildItem -LiteralPath $repoDestination -Directory -Recurse -Force |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".git", "wandb", "cache") } |
    Sort-Object { $_.FullName.Length } -Descending |
    Remove-Item -Recurse -Force

$manifestRows = foreach ($file in Get-ChildItem -LiteralPath $staging -File -Recurse | Sort-Object FullName) {
    $relative = [IO.Path]::GetRelativePath($staging, $file.FullName).Replace('\', '/')
    if ($relative -match '(^|/)(\.env($|\.)|id_rsa|id_ed25519|wireguard|wg0|api[_-]?key)') { throw "Secret-like file selected: $relative" }
    [pscustomobject]@{ path = $relative; size = $file.Length; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant() }
}
$manifest = [ordered]@{
    schema_version = "sac-migration-manifest-v1"
    source_archive = [ordered]@{ size = $archiveInfo.Length; sha256 = $archiveHash }
    file_count = @($manifestRows).Count
    files = @($manifestRows)
}
$manifestPath = Join-Path $staging "migration-manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
$bundle = Join-Path $outputPath "sac-qutab-continuation.tar"
& tar -cf $bundle -C $staging .
if ($LASTEXITCODE -ne 0) { throw "Bundle creation failed" }
$bundleHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash.ToLowerInvariant()
"$bundleHash  sac-qutab-continuation.tar" | Set-Content -LiteralPath "$bundle.sha256" -Encoding ascii

$input = [IO.File]::OpenRead($bundle)
try {
    $buffer = New-Object byte[] (4MB)
    $partIndex = 0
    while ($input.Position -lt $input.Length) {
        $partPath = Join-Path $outputPath ("sac-qutab-continuation.tar.part{0:D3}" -f $partIndex)
        $part = [IO.File]::Create($partPath)
        try {
            $remaining = [Math]::Min($PartBytes, $input.Length - $input.Position)
            while ($remaining -gt 0) {
                $read = $input.Read($buffer, 0, [int][Math]::Min($buffer.Length, $remaining))
                if ($read -le 0) { throw "Unexpected end of bundle while splitting" }
                $part.Write($buffer, 0, $read)
                $remaining -= $read
            }
        } finally { $part.Dispose() }
        $partIndex++
    }
} finally { $input.Dispose() }

[pscustomobject]@{ bundle = $bundle; sha256 = $bundleHash; parts = $partIndex; files = @($manifestRows).Count } | ConvertTo-Json
