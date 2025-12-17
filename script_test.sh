RESOURCE_GROUP="rg-mlops-bank-churn"

CONTAINER_APP_NAME="bank-churn" 

APP_URL=$(az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv | tr -d '\r\n' | xargs)

# 2. Vérifier l'URL proprement
echo "URL nettoyée: '$APP_URL'"
echo "Longueur: ${#APP_URL}"

# 3. Test avec l'URL complète
FULL_URL="https://${APP_URL}/predict"
echo "URL complète: $FULL_URL"

# 4. Test de prédiction
curl -X POST "$FULL_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000,
    "Geography_Germany": 0,
    "Geography_Spain": 1
  }'

echo ""