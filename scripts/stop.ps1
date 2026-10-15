# LLM Router - Daemon Stopper
param()

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Stopping LLM Router background daemon..." -ForegroundColor Cyan

$stopped = $false
$pidFile = Join-Path $root "router.pid"

if (Test-Path $pidFile) {
    $pidNum = (Get-Content $pidFile -Raw).Trim()
    if ($pidNum) {
        try {
            Stop-Process -Id ([int]$pidNum) -Force -ErrorAction SilentlyContinue
            Write-Host ("  >> Stopped process PID " + $pidNum) -ForegroundColor Green
            $stopped = $true
        } catch {}
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

# Also ensure port 8080 listeners are cleaned up
try {
    $conns = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        foreach ($c in $conns) {
            $p = $c.OwningProcess
            if ($p -and $p -ne 0) {
                try {
                    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                    Write-Host ("  >> Terminated listener on port 8080 (PID " + $p + ")") -ForegroundColor Green
                    $stopped = $true
                } catch {}
            }
        }
    }
} catch {
    # Fallback for older netstat
    $port8080 = netstat -ano | Select-String "LISTENING" | Select-String ":8080"
    if ($port8080) {
        $pids = ($port8080 | ForEach-Object { ($_ -split '\s+')[-1] }) | Select-Object -Unique
        foreach ($p in $pids) {
            try {
                Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue
                Write-Host ("  >> Terminated listener on port 8080 (PID " + $p + ")") -ForegroundColor Green
                $stopped = $true
            } catch {}
        }
    }
}

if ($stopped) {
    Write-Host ">> LLM Router daemon has been stopped." -ForegroundColor Green
} else {
    Write-Host ">> No running LLM Router process found." -ForegroundColor Yellow
}
