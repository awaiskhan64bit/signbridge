# Creates a Desktop shortcut for SignBridge.
#
# Written as its own script rather than inline in the batch file because a
# multi line PowerShell command inside a .bat needs escaping at every level, and
# the failure when that goes wrong is an empty command that reports nothing.

param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [string]$ShortcutName = "SignBridge"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    Write-Output "shortcut: interpreter not found at $Python, skipping"
    exit 1
}

$desktop = [Environment]::GetFolderPath("Desktop")
$target = Join-Path $desktop "$ShortcutName.lnk"

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($target)

$link.TargetPath = $Python
$link.Arguments = "-m signbridge"
$link.WorkingDirectory = $WorkDir
$link.Description = "SignBridge, offline sign language to speech"
$link.IconLocation = "$Python,0"

# A console window behind the app is noise on a desktop shortcut. The app is a
# GUI, so hide it, but keep stderr so a crash is still diagnosable.
$link.WindowStyle = 7

$link.Save()

if (Test-Path $target) {
    Write-Output "shortcut: created $target"
} else {
    Write-Output "shortcut: could not write $target"
    exit 1
}