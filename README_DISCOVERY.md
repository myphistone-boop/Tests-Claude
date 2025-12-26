# Module de Découverte de Vidéos Virales YouTube

## 📋 Description

Ce module analyse les vidéos trending sur YouTube (France et USA) et sélectionne automatiquement les 10 meilleures vidéos pour créer des shorts viraux :
- **5 vidéos françaises** (région FR)
- **5 vidéos anglaises** (région US)

## 🎯 Critères de Sélection

### Filtres Automatiques
- ✅ Durée maximale : **15 minutes** (pour limiter les coûts API)
- ✅ Présence de musique : **acceptée** (pas de filtrage)
- ✅ Vidéos publiques et accessibles

### Score de Viralité
Le système calcule un score de viralité (0-100) basé sur :

1. **Vitesse de propagation (50%)** : Vues par heure depuis publication
2. **Engagement (40%)** :
   - Ratio likes/vues (engagement positif)
   - Ratio commentaires/vues (engagement conversationnel)
3. **Fraîcheur (10%)** : Bonus pour les vidéos récentes (<24h)

## 🚀 Installation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Configurer l'API YouTube

1. Créez un projet sur [Google Cloud Console](https://console.cloud.google.com/)
2. Activez l'API **YouTube Data API v3**
3. Créez une clé API dans "Identifiants"
4. Copiez `.env.example` vers `.env` :

```bash
cp .env.example .env
```

5. Ajoutez votre clé API YouTube dans `.env` :

```env
YOUTUBE_API_KEY=votre-cle-api-youtube-ici
```

## 📊 Utilisation

### Lancer la découverte

```bash
python discover_videos.py
```

### Processus (5 étapes)

```
ÉTAPE 1/5 : DÉCOUVERTE VIDÉOS FRANÇAISES
  🔍 Recherche des vidéos trending FR...
  📡 API YouTube FR: [████████] 100%
  🔎 Filtrage FR: [████████] 50/50
  ✅ 42 vidéos valides (≤15 min)

ÉTAPE 2/5 : DÉCOUVERTE VIDÉOS ANGLAISES
  🔍 Recherche des vidéos trending EN...
  📡 API YouTube EN: [████████] 100%
  🔎 Filtrage EN: [████████] 50/50
  ✅ 45 vidéos valides (≤15 min)

ÉTAPE 3/5 : SÉLECTION TOP 5 FRANÇAISES
  🎯 Calcul des scores de viralité pour 42 vidéos...
  📊 Scoring: [████████] 42/42
  ✅ Top 5 sélectionnées

ÉTAPE 4/5 : SÉLECTION TOP 5 ANGLAISES
  🎯 Calcul des scores de viralité pour 45 vidéos...
  📊 Scoring: [████████] 45/45
  ✅ Top 5 sélectionnées

ÉTAPE 5/5 : SAUVEGARDE DES RÉSULTATS
  ✅ Résultats sauvegardés: selected_videos/selected_videos_20231215_143022.json
```

## 📁 Fichier de Sortie

Les résultats sont sauvegardés dans `selected_videos/selected_videos_YYYYMMDD_HHMMSS.json` :

```json
{
  "timestamp": "20231215_143022",
  "date": "2023-12-15T14:30:22.123456",
  "total_selected": 10,
  "french_count": 5,
  "english_count": 5,
  "videos": [
    {
      "video_id": "dQw4w9WgXcQ",
      "title": "Titre de la vidéo",
      "channel_title": "Nom de la chaîne",
      "duration_seconds": 213,
      "view_count": 1234567,
      "like_count": 98765,
      "comment_count": 4321,
      "published_at": "2023-12-15T10:00:00Z",
      "language": "fr",
      "virality_score": 87.45,
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }
  ]
}
```

## 💰 Coûts API

### YouTube Data API v3

- **Quota gratuit** : 10 000 unités/jour
- **Coût par requête** :
  - Liste trending (50 vidéos) : 1 unité
  - Total pour 1 découverte : **2 unités** (FR + EN)

Vous pouvez lancer la découverte **5 000 fois par jour** gratuitement.

## 🔧 Configuration Avancée

### Modifier les paramètres dans `discover_videos.py`

```python
class YouTubeTrendingAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        # Paramètres modifiables
        self.max_duration_seconds = 15 * 60  # 15 minutes
        self.max_results_per_language = 50   # Nombre de vidéos à analyser
```

### Changer les régions

```python
# Français (France)
fr_videos = analyzer.get_trending_videos(region_code='FR', language='fr')

# Français (Canada)
fr_videos = analyzer.get_trending_videos(region_code='CA', language='fr')

# Anglais (UK)
en_videos = analyzer.get_trending_videos(region_code='GB', language='en')
```

## 📈 Prochaines Étapes

Ce module fait partie d'un système complet d'automatisation :

1. ✅ **Découverte** : Sélection automatique des vidéos (ce module)
2. ⏳ **Traitement** : Création des shorts avec sous-titres (existant)
3. ⏳ **Publication** : Upload automatique YouTube + TikTok
4. ⏳ **Analytics** : Suivi des performances
5. ⏳ **Optimisation** : Machine learning sur les tendances

## 🐛 Dépannage

### Erreur "YOUTUBE_API_KEY non trouvée"

Vérifiez que :
1. Le fichier `.env` existe
2. La clé `YOUTUBE_API_KEY` est présente dans `.env`
3. Le fichier `.env` est dans le même dossier que `discover_videos.py`

### Erreur API 403 (Quota dépassé)

Vous avez épuisé votre quota quotidien (10 000 unités). Attendez 24h ou :
1. Créez un nouveau projet Google Cloud
2. Utilisez une nouvelle clé API

### Aucune vidéo valide trouvée

Les vidéos trending dépassent toutes 15 minutes. Augmentez la limite :

```python
self.max_duration_seconds = 30 * 60  # 30 minutes
```

## 📞 Support

Pour toute question ou problème, consultez la documentation YouTube Data API :
- [Guide officiel](https://developers.google.com/youtube/v3)
- [Quota et limites](https://developers.google.com/youtube/v3/getting-started#quota)
