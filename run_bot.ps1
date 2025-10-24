# run_bot.ps1
# Encodage UTF-8 pour éviter les caractères bizarres
chcp 65001 > $null

# Dossier courant = dossier du script
Set-Location $PSScriptRoot

# Dossier logs
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Horodatage
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs\run_$ts.log"

# ---- SÉCURITÉ : forcer DRY RUN pour ce run seulement ----
#$env:FORCE_DRYRUN = "1" 

Write-Host "=== Lancement du bot (FORCE_DRYRUN=1) à $ts ==="
python -u -m strategies.Ibkr_fx_intraday *>&1 | Tee-Object -FilePath $logFile


# Nettoyage de la variable d'env (n'affecte pas les autres sessions)
Remove-Item Env:\FORCE_DRYRUN -ErrorAction SilentlyContinue

Write-Host "=== Log sauvegardé -> $logFile ==="
