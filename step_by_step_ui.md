# Guide Complet - Déploiement Azure Container Apps via l'Interface Graphique Azure

## **Objectif**
Reproduire EXACTEMENT le script Bash fourni en utilisant UNIQUEMENT l'interface graphique Azure Portal.

## **Prérequis**
1. Compte Azure avec abonnement actif
2. Accès à [portal.azure.com](https://portal.azure.com)
3. Dockerfile et code de l'application `bank-churn-api` prêts localement

---

## **ÉTAPE 0: Connexion Azure**
1. **Connectez-vous** à [portal.azure.com](https://portal.azure.com)
2. **Vérifiez votre abonnement** :
   - En haut à droite → Cliquez sur votre profil
   - "Changer de répertoire" si besoin
   - L'abonnement actif s'affiche dans le panneau latéral gauche

---

## **ÉTAPE 1: Vérifier/Créer les Fournisseurs (Providers)**
⚠️ **Cette étape n'est pas faisable dans le portail**  
Les providers s'enregistrent automatiquement lors de la première utilisation du service.  
**Alternative** : Utilisez Azure Cloud Shell (Bash) pour cette partie uniquement :

```bash
# Dans Azure Cloud Shell (icône >_ en haut du portail)
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.Web --wait
az provider register --namespace Microsoft.OperationalInsights --wait
```

---

## **ÉTAPE 2: Groupe de Ressources**
1. **Recherchez** "Groupes de ressources" dans la barre de recherche
2. **Cliquez** sur "+ Créer"
3. **Remplissez** :
   - Abonnement : Votre abonnement
   - Groupe de ressources : `rg-MLopsyy`
   - Région : `France Central`
4. **Cliquez** sur "Vérifier + créer" puis "Créer"
5. **Attendez** le déploiement (≈30 secondes)

---

## **ÉTAPE 3: Container Registry (ACR)**
### **3.1 Création ACR**
1. **Recherchez** "Registres de conteneurs"
2. **Cliquez** sur "+ Créer"
3. **Onglet "Général"** :
   - Groupe de ressources : `rg-MLopsyy`
   - Nom du registre : `acrmlops[VOTRE_USERNAME][TIMESTAMP]`  
     *Ex: acrmlopsjean1648826400*
   - Emplacement : `France Central`
   - SKU : `De base`
4. **Onglet "Authentification"** :
   - ✅ Utilisateur administrateur → ACTIVÉ
5. **Cliquez** sur "Vérifier + créer" puis "Créer"

### **3.2 Fallback si France Central bloqué**
Si erreur de stratégie :
1. **Recommencez** l'étape 3.1
2. **Changez** l'emplacement : `West Europe`
3. **Notez** la nouvelle région pour les étapes suivantes

---

## **ÉTAPE 4: Build et Push de l'Image**
### **4.1 Préparer localement**
```bash
# Sur VOTRE machine locale (pas dans le portail)
cd /chemin/vers/votre/projet

# Build l'image
docker build -t bank-churn-api:v1 .

# Tag avec ACR
docker tag bank-churn-api:v1 acrmlopsjean1648826400.azurecr.io/bank-churn-api:v1
docker tag bank-churn-api:v1 acrmlopsjean1648826400.azurecr.io/bank-churn-api:latest
```

### **4.2 Push vers ACR**
#### **Option A: Via Azure CLI local**
```bash
# Login ACR
az acr login --name acrmlopsjean1648826400

# Push images
docker push acrmlopsjean1648826400.azurecr.io/bank-churn-api:v1
docker push acrmlopsjean1648826400.azurecr.io/bank-churn-api:latest
```

#### **Option B: Via Portail Azure**
1. **Allez** dans votre ACR créé
2. **Menu gauche** → "Services" → "Tâches"
3. **Cliquez** sur "+ Tâche"
4. **Configurez** :
   - Type de tâche : Tâche rapide
   - Platform : Linux
   - Emplacement : Même que l'ACR
   - Source du code : "Context local"
   - Uploader votre code ZIP ou Dockerfile
5. **Exécutez** la tâche

---

## **ÉTAPE 5: Log Analytics Workspace**
1. **Recherchez** "Espaces de travail Log Analytics"
2. **Cliquez** sur "+ Créer"
3. **Remplissez** :
   - Groupe de ressources : `rg-MLopsyy`
   - Nom : `law-mlops-[VOTRE_USERNAME]-[RANDOM]`  
     *Ex: law-mlops-jean-12345*
   - Région : Même que l'ACR (France Central ou West Europe)
4. **Cliquez** sur "Vérifier + créer" puis "Créer"
5. **Notez** :
   - **ID de l'espace de travail** (customerId)
   - **Clé primaire** (primarySharedKey)

---

## **ÉTAPE 6: Container Apps Environment**
1. **Recherchez** "Environnements Container Apps"
2. **Cliquez** sur "+ Créer"
3. **Onglet "Général"** :
   - Nom de l'environnement : `env-mlops-workshop`
   - Groupe de ressources : `rg-MLopsyy`
   - Zone : Même région que l'ACR
4. **Onglet "Surveillance"** :
   - ✅ Activer la surveillance Log Analytics
   - Espace de travail Log Analytics : Sélectionnez celui créé à l'étape 5
5. **Cliquez** sur "Vérifier + créer" puis "Créer"

---

## **ÉTAPE 7: Container App (Application)**
### **7.1 Création**
1. **Recherchez** "Container Apps"
2. **Cliquez** sur "+ Créer"
3. **Onglet "Général"** :
   - Nom de l'application : `bank-churn-api`
   - Groupe de ressources : `rg-MLopsyy`
   - Environnement Container Apps : `env-mlops-workshop` (créé précédemment)

### **7.2 Onglet "Application"**
1. **Section "Image"** :
   - Source de l'image : "Azure Container Registry"
   - Registre : Sélectionnez votre ACR
   - Image : `bank-churn-api`
   - Étiquette : `v1`
2. **Authentification ACR** :
   - Type d'authentification : "Informations d'identification de l'administrateur"
   - Nom d'utilisateur/Password : Récupérez-les dans ACR → "Clés d'accès"

### **7.3 Onglet "Ingress"**
1. **Trafic entrant** : ✅ Activé
2. **Visibilité du trafic entrant** : Externe
3. **Port cible** : `8000`

### **7.4 Onglet "Mise à l'échelle"**
1. **Mode de mise à l'échelle** : "Aucune mise à l'échelle automatique"
2. **Nombre minimal de réplicas** : `1`
3. **Nombre maximal de réplicas** : `1`

### **7.5 Finalisation**
- **Cliquez** sur "Vérifier + créer" puis "Créer"
- **Attendez** le déploiement (≈2-3 minutes)

---

## **ÉTAPE 8: Récupérer l'URL**
1. **Allez** sur votre Container App `bank-churn-api`
2. **Menu gauche** → "Application"
3. **Cherchez** "URL de l'application"
4. **Copiez** l'URL (format : `https://bank-churn-api.randomtext.region.azurecontainerapps.io`)

---

## **ÉTAPE 9: Tests**
1. **Ouvrez** un navigateur
2. **Testez** :
   - **Health** : `https://[VOTRE-URL]/health`
   - **Documentation** : `https://[VOTRE-URL]/docs`
   - **Swagger UI** : `https://[VOTRE-URL]/redoc`

---

## **Vérification Finale**
Comparez avec le script Bash :

| Élément | Script Bash | Interface Graphique |
|---------|------------|-------------------|
| Resource Group | `rg-MLopsyy` (France Central) | ✅ Identique |
| ACR | Nom unique avec timestamp | ✅ Identique |
| Fallback location | West Europe si blocage | ✅ Géré manuellement |
| Log Analytics | Créé avec nom aléatoire | ✅ Identique |
| Environment | `env-mlops-workshop` | ✅ Identique |
| Container App | `bank-churn-api` port 8000 | ✅ Identique |
| Image | `bank-churn-api:v1` | ✅ Identique |
| Réplicas | min=1, max=1 | ✅ Identique |
| Ingress | Externe | ✅ Identique |

---

## **Points d'Attention**
1. **Timestamp dans ACR** : Dans le portail, générez-le manuellement (ex: `date +%s` dans Cloud Shell)
2. **Authentification ACR** : Récupérez les credentials dans ACR → "Clés d'accès"
3. **Variables d'environnement** : Si votre app en a besoin, ajoutez-les dans l'onglet "Variables d'environnement" du Container App
4. **Logs** : Les logs sont automatiquement envoyés à Log Analytics

---

## **Résumé des URLs**
- **Portail Azure** : https://portal.azure.com
- **Votre API** : `https://bank-churn-api.[...].azurecontainerapps.io`
- **Health check** : `/health`
- **Documentation** : `/docs` (Swagger)
- **ACR** : `acrmlopsjean1648826400.azurecr.io`

---

**Durée totale** : ≈15-20 minutes via l'interface graphique  
**Coût estimé** : ~5-10€/mois (ACR Basic + Container App)