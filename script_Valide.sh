#!/usr/bin/env bash
set -euo pipefail

#################################
# CONFIGURATION
#################################
RESOURCE_GROUP="rg-MLopsyy"
CONTAINER_APP_NAME="bank-churn-api"
CONTAINERAPPS_ENV="env-mlops-workshop"
IMAGE_NAME="bank-churn-api"
IMAGE_TAG="v1"
TARGET_PORT=8000

# Liste des régions à tester par ordre de préférence
REGION_PRIORITY=("westeurope" "northeurope" "francecentral" "uksouth" "eastus" "westus")
MAX_RETRIES=3  # Nombre maximum de tentatives

#################################
# FONCTIONS UTILITAIRES
#################################
detect_available_region() {
    local selected_region=""
    
    for region in "${REGION_PRIORITY[@]}"; do
        echo "Vérification de la disponibilité de la région: $region"
        
        # Vérifie si la région supporte Container Apps
        if az provider show --namespace "Microsoft.App" --query "resourceTypes[?resourceType=='environments'].locations[]" -o tsv | grep -iq "$region"; then
            echo "✅ Région $region supporte Container Apps"
            selected_region="$region"
            break
        else
            echo "⚠️ Région $region non disponible pour Container Apps"
        fi
    done
    
    if [ -z "$selected_region" ]; then
        echo "❌ Aucune région disponible dans la liste. Utilisation de westeurope par défaut."
        selected_region="westeurope"
    fi
    
    echo "📍 Région sélectionnée: $selected_region"
    echo "$selected_region"
}

create_acr_with_retry() {
    local rg="$1"
    local acr_name="$2"
    local region="$3"
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        echo "Tentative $attempt/$MAX_RETRIES pour créer ACR en $region..."
        
        if az acr create \
            --resource-group "$rg" \
            --name "$acr_name" \
            --sku Basic \
            --admin-enabled true \
            --location "$region" >/dev/null 2>&1; then
            echo "✅ ACR créé avec succès en $region"
            return 0
        fi
        
        echo "⚠️ Échec tentative $attempt, attente avant réessai..."
        sleep 5
        ((attempt++))
    done
    
    echo "❌ Échec après $MAX_RETRIES tentatives"
    return 1
}

# Fonction pour nettoyer les retours chariot
clean_output() {
    echo "$1" | tr -d '\r\n'
}

#################################
# DÉBUT DU SCRIPT
#################################

# 0) Contexte Azure
echo "Vérification du contexte Azure..."
az account show --query "{name:name}" -o tsv

# 0.5) Extension Container App
echo "Vérification/installation de l'extension containerapp..."
if ! az extension list --query "[?name=='containerapp'].name" -o tsv | grep -q containerapp; then
    echo "Installation de l'extension containerapp..."
    az extension add --name containerapp --yes
else
    echo "✅ Extension containerapp déjà installée"
fi

# 1) Détection de région disponible
echo "Détection de région disponible..."
LOCATION=$(detect_available_region)
FALLBACK_LOCATION="$LOCATION"  # Même région pour fallback
echo "📍 Région finale sélectionnée: $LOCATION"

# 2) Providers nécessaires
echo "Enregistrement des providers..."
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

# 3) Resource Group
echo "Création/validation du groupe de ressources..."
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" >/dev/null || true
echo "✅ RG OK: $RESOURCE_GROUP"

# 4) Création ACR avec nom unique
ACR_NAME="acrmlops$(whoami)$(date +%s | tail -c 6)"
echo "Création du Container Registry (ACR): $ACR_NAME"

if ! create_acr_with_retry "$RESOURCE_GROUP" "$ACR_NAME" "$LOCATION"; then
    # Fallback: essayer une autre région
    echo "⚠️ Fallback: essai dans une région alternative..."
    for alt_region in "${REGION_PRIORITY[@]}"; do
        if [ "$alt_region" != "$LOCATION" ]; then
            if create_acr_with_retry "$RESOURCE_GROUP" "$ACR_NAME" "$alt_region"; then
                LOCATION="$alt_region"
                echo "📍 Région changée à: $LOCATION"
                break
            fi
        fi
    done
fi

# NETTOYAGE CRITIQUE : retirer \r de la sortie Azure CLI
ACR_LOGIN_SERVER=$(clean_output "$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)")
echo "✅ ACR créé: $ACR_NAME (login: $ACR_LOGIN_SERVER)"

# 5) Build et push de l'image
echo "Build + Tag + Push..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
echo "✅ Image pushée dans ACR"

# 6) Log Analytics
LAW_NAME="law-mlops-$(whoami)-$RANDOM"
echo "Création Log Analytics: $LAW_NAME"
az monitor log-analytics workspace create -g "$RESOURCE_GROUP" -n "$LAW_NAME" -l "$LOCATION" >/dev/null

LAW_ID=$(clean_output "$(az monitor log-analytics workspace show -g "$RESOURCE_GROUP" -n "$LAW_NAME" --query customerId -o tsv)")
LAW_KEY=$(clean_output "$(az monitor log-analytics workspace get-shared-keys -g "$RESOURCE_GROUP" -n "$LAW_NAME" --query primarySharedKey -o tsv)")
echo "✅ Log Analytics OK"

# 7) Environment Container Apps
echo "Création/validation Container Apps Environment: $CONTAINERAPPS_ENV"
if ! az containerapp env show -n "$CONTAINERAPPS_ENV" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
    echo "Création de l'environment..."
    az containerapp env create \
        -n "$CONTAINERAPPS_ENV" \
        -g "$RESOURCE_GROUP" \
        -l "$LOCATION" \
        --logs-workspace-id "$LAW_ID" \
        --logs-workspace-key "$LAW_KEY" >/dev/null
fi
echo "✅ Environment OK"

# 8) Déploiement Container App
ACR_USER=$(clean_output "$(az acr credential show -n "$ACR_NAME" --query username -o tsv)")
ACR_PASS=$(clean_output "$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)")
IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "Déploiement Container App: $CONTAINER_APP_NAME"
if az containerapp show -n "$CONTAINER_APP_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
    az containerapp update \
        -n "$CONTAINER_APP_NAME" \
        -g "$RESOURCE_GROUP" \
        --image "$IMAGE" \
        --registry-server "$ACR_LOGIN_SERVER" \
        --registry-username "$ACR_USER" \
        --registry-password "$ACR_PASS" >/dev/null
else
    az containerapp create \
        -n "$CONTAINER_APP_NAME" \
        -g "$RESOURCE_GROUP" \
        --environment "$CONTAINERAPPS_ENV" \
        --image "$IMAGE" \
        --ingress external \
        --target-port "$TARGET_PORT" \
        --registry-server "$ACR_LOGIN_SERVER" \
        --registry-username "$ACR_USER" \
        --registry-password "$ACR_PASS" \
        --min-replicas 1 \
        --max-replicas 1 >/dev/null
fi
echo "✅ Container App OK"

# 9) Récupération URL
APP_URL=$(clean_output "$(az containerapp show -n "$CONTAINER_APP_NAME" -g "$RESOURCE_GROUP" \
    --query properties.configuration.ingress.fqdn -o tsv)")

echo ""
echo "=========================================="
echo "✅ DÉPLOIEMENT RÉUSSI"
echo "=========================================="
echo "Région      : $LOCATION"
echo "ACR         : $ACR_NAME"
echo "Environment : $CONTAINERAPPS_ENV"
echo "API URL     : https://$APP_URL"
echo "Health      : https://$APP_URL/health"
echo "Docs        : https://$APP_URL/docs"
echo "=========================================="