# deploy.ps1

# Variables (MODIFIEZ avec vos valeurs)
$RESOURCE_GROUP = "rg-mlops1"
$LOCATION = "westeurope"
$TIMESTAMP = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$USERNAME = $env:USERNAME
$ACR_NAME = "acrmlops$USERNAME$TIMESTAMP"  # Doit être unique globalement
$CONTAINER_APP_NAME = "app-churn-api"

Write-Host "Début du déploiement..." -ForegroundColor Green
Write-Host "Nom ACR généré: $ACR_NAME" -ForegroundColor Yellow

# Vérifier si connecté à Azure
try {
    $account = az account show 2>$null | ConvertFrom-Json
    if (-not $account) {
        Write-Host "Connexion à Azure requise..." -ForegroundColor Yellow
        az login
    }
}
catch {
    Write-Host "Connexion à Azure requise..." -ForegroundColor Yellow
    az login
}

# Création du groupe de ressources
Write-Host "Création du groupe de ressources..." -ForegroundColor Cyan
az group create `
  --name $RESOURCE_GROUP `
  --location $LOCATION

Write-Host "Groupe de ressources créé : $RESOURCE_GROUP" -ForegroundColor Green