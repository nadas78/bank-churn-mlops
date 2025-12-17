# Variables (MODIFIEZ avec vos valeurs)
RESOURCE_GROUP="rg-mlops"
LOCATION="westeurope"
ACR_NAME="acrmlops$(whoami)$(date +%s)"  # Doit être unique globalement
CONTAINER_APP_NAME="bank-churn-api"
CONTAINERAPPS_ENV="env-mlops-workshop"

# Vérifier et définir le contexte Azure (IMPORTANT)
echo "Vérification du contexte Azure..."
az account show --query "{name:name, cloudName:cloudName}" || az login

# Enregistrer le fournisseur de ressources pour Microsoft.ContainerRegistry si ce n'est pas déjà fait
echo "Enregistrement du fournisseur de ressources Microsoft.ContainerRegistry..."
az provider register --namespace Microsoft.ContainerRegistry

# Vérifier que le fournisseur est bien enregistré
az provider show --namespace Microsoft.ContainerRegistry --query "registrationState"

# Création du groupe de ressources
echo "Création du groupe de ressources..."
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

echo "✅ Groupe de ressources créé : $RESOURCE_GROUP"

# Création du Container Registry
echo "Création du Container Registry..."
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true \
  --location $LOCATION

echo "✅ Container Registry créé : $ACR_NAME"

# Se connecter au registry
echo "Connexion au registry..."
az acr login --name $ACR_NAME

# Vérification de la connexion au registry
echo "Vérification de la connexion..."
az acr check-health --name $ACR_NAME 

