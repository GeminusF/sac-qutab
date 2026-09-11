[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Archive,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [long]$PartBytes = 512MB,

    [switch]$InventoryOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedArchiveBytes = 14067722240
$expectedArchiveSha256 = "cb9ef494946f4ca43a883768904df234fd2f57b36e46c3cdc61f29c0957337f2"
$expectedCacheFiles = 129600
$roles = @("validation", "calibration", "final_test")

$archivePath = [IO.Path]::GetFullPath($Archive)
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Source archive not found: $archivePath"
}
$archiveInfo = Get-Item -LiteralPath $archivePath
if ($archiveInfo.Length -ne $expectedArchiveBytes) {
    throw "Unexpected source archive size: $($archiveInfo.Length)"
}
$archiveSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
if ($archiveSha256 -ne $expectedArchiveSha256) {
    throw "Source archive SHA-256 mismatch: $archiveSha256"
}

$entries = @(& tar -tf $archivePath)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to list source archive"
}
foreach ($entry in $entries) {
    $normalized = $entry.Replace('\', '/')
    if ($normalized.StartsWith('/') -or $normalized -match '^[A-Za-z]:' -or $normalized.Split('/') -contains '..') {
        throw "Unsafe archive entry: $entry"
    }
}

$selected = @($entries | Where-Object {
    $normalized = $_.Replace('\', '/')
    -not $normalized.EndsWith('/') -and (
        $normalized -match '^artifacts/counterfactuals/cache/[^/]+\.npy$' -or
        $normalized -match '^artifacts/counterfactuals/(validation|calibration|final_test)(\.summary)?\.jsonl$' -or
        $normalized -match '^artifacts/counterfactuals/(validation|calibration|final_test)\.summary\.json$'
    )
})
$cacheEntries = @($selected | Where-Object { $_.Replace('\', '/') -match '^artifacts/counterfactuals/cache/[^/]+\.npy$' })
if ($cacheEntries.Count -ne $expectedCacheFiles) {
    throw "Expected $expectedCacheFiles cache tensors, found $($cacheEntries.Count)"
}
foreach ($role in $roles) {
    foreach ($suffix in @(".jsonl", ".summary.json")) {
        $required = "artifacts/counterfactuals/$role$suffix"
        if ($selected -notcontains $required) {
            throw "Required frozen file is missing from source archive: $required"
        }
    }
}

if ($InventoryOnly) {
    [pscustomobject]@{
        source_archive = $archivePath
        source_sha256 = $archiveSha256
        cache_files = $cacheEntries.Count
        selected_files = $selected.Count
    } | ConvertTo-Json
    exit 0
}

$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $outputPath) {
    if ((Get-ChildItem -LiteralPath $outputPath -Force | Measure-Object).Count -gt 0) {
        throw "Output directory must be empty: $outputPath"
    }
} else {
    New-Item -ItemType Directory -Path $outputPath | Out-Null
}

$staging = Join-Path $outputPath "staging"
New-Item -ItemType Directory -Path $staging | Out-Null
$allowlistPath = Join-Path $outputPath "cache-archive-allowlist.txt"
$selected | Set-Content -LiteralPath $allowlistPath -Encoding utf8
& tar -xf $archivePath -C $staging -T $allowlistPath
if ($LASTEXITCODE -ne 0) {
    throw "Cache allowlist extraction failed"
}

$expectedHashesPath = Join-Path $staging "cache-transfer-files.sha256"
$writer = [IO.StreamWriter]::new($expectedHashesPath, $false, [Text.UTF8Encoding]::new($false))
$seenCachePaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
try {
    foreach ($role in $roles) {
        $manifestRelative = "artifacts/counterfactuals/$role.jsonl"
        $manifestPath = Join-Path $staging $manifestRelative
        foreach ($line in [IO.File]::ReadLines($manifestPath)) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $row = $line | ConvertFrom-Json
            $cacheRelative = "artifacts/" + [string]$row.cache_relpath
            $cacheHash = [string]$row.cache_sha256
            if ($cacheHash -notmatch '^[0-9a-f]{64}$') {
                throw "Invalid frozen cache SHA-256 in $manifestRelative"
            }
            if (-not $seenCachePaths.Add($cacheRelative)) {
                throw "Duplicate cache path across frozen manifests: $cacheRelative"
            }
            $writer.WriteLine("$cacheHash  $cacheRelative")
        }
        foreach ($relative in @($manifestRelative, "artifacts/counterfactuals/$role.summary.json")) {
            $filePath = Join-Path $staging $relative
            $fileHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $filePath).Hash.ToLowerInvariant()
            $writer.WriteLine("$fileHash  $relative")
        }
    }
} finally {
    $writer.Dispose()
}
if ($seenCachePaths.Count -ne $expectedCacheFiles) {
    throw "Frozen manifests reference $($seenCachePaths.Count) unique cache files; expected $expectedCacheFiles"
}

$bundle = Join-Path $outputPath "sac-qutab-cache-transfer.tar"
& tar -cf $bundle -C $staging .
if ($LASTEXITCODE -ne 0) {
    throw "Cache transfer archive creation failed"
}
$bundleHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash.ToLowerInvariant()
"$bundleHash  sac-qutab-cache-transfer.tar" | Set-Content -LiteralPath "$bundle.sha256" -Encoding ascii

$input = [IO.File]::OpenRead($bundle)
try {
    $buffer = New-Object byte[] (4MB)
    $partIndex = 0
    while ($input.Position -lt $input.Length) {
        $partPath = Join-Path $outputPath ("sac-qutab-cache-transfer.tar.part{0:D3}" -f $partIndex)
        $part = [IO.File]::Create($partPath)
        try {
            $remaining = [Math]::Min($PartBytes, $input.Length - $input.Position)
            while ($remaining -gt 0) {
                $read = $input.Read($buffer, 0, [int][Math]::Min($buffer.Length, $remaining))
                if ($read -le 0) { throw "Unexpected end of cache transfer archive" }
                $part.Write($buffer, 0, $read)
                $remaining -= $read
            }
        } finally {
            $part.Dispose()
        }
        $partIndex++
    }
} finally {
    $input.Dispose()
}

[pscustomobject]@{
    bundle = $bundle
    sha256 = $bundleHash
    size = (Get-Item -LiteralPath $bundle).Length
    parts = $partIndex
    cache_files = $expectedCacheFiles
    selected_files = $selected.Count
} | ConvertTo-Json
