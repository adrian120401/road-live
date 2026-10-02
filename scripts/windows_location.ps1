# Windows PowerShell 5.1 / .NET Framework. One persistent native location watcher.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$watcher = $null
try {
    Add-Type -AssemblyName System.Device
    $watcher = New-Object System.Device.Location.GeoCoordinateWatcher([System.Device.Location.GeoPositionAccuracy]::High)
    $watcher.MovementThreshold = 0
    $watcher.Start()
    while ($true) {
        $position = $watcher.Position
        $coordinate = $position.Location
        $row = @{ status = $watcher.Status.ToString(); permission = $watcher.Permission.ToString() }
        if (-not $coordinate.IsUnknown) {
            $row.latitude = $coordinate.Latitude
            $row.longitude = $coordinate.Longitude
            $row.accuracy_m = $coordinate.HorizontalAccuracy
            $row.timestamp = $position.Timestamp.ToUniversalTime().ToString('o')
        }
        [Console]::WriteLine(($row | ConvertTo-Json -Compress))
        Start-Sleep -Milliseconds 500
    }
} catch {
    [Console]::WriteLine((@{ status = 'Error'; error = $_.Exception.Message } | ConvertTo-Json -Compress))
    exit 1
} finally {
    if ($null -ne $watcher) { $watcher.Stop(); $watcher.Dispose() }
}
