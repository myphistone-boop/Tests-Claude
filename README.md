# 🎥 Générateur de Sous-titres YouTube Style TikTok

Créez automatiquement des vidéos YouTube avec des sous-titres stylisés mot par mot, comme sur TikTok !

## ✨ Fonctionnalités

- 📥 Téléchargement de vidéos YouTube
- 🎵 Extraction automatique de l'audio
- 🎙️ Transcription avec timestamps mot par mot (Whisper API)
- 🎬 Génération de sous-titres style TikTok (mots qui apparaissent un par un)
- 🌍 Support Français et Anglais
- 💯 Pipeline 100% automatisé en un seul script

## 📋 Prérequis

- **Python 3.7 ou supérieur**
- **VS Code** (ou tout autre éditeur de code)
- **Clé API OpenAI** (pour Whisper)

## 🔑 Obtenir votre clé API OpenAI

1. Allez sur : https://platform.openai.com/api-keys
2. Créez un compte (ou connectez-vous)
3. Cliquez sur "Create new secret key"
4. Copiez la clé (format : `sk-...`)
5. **Important** : Ajoutez un moyen de paiement sur votre compte

**💰 Coût** : ~0.006$/minute d'audio (ex: vidéo de 10 min = $0.06)

## 🚀 Installation

### 1. Cloner le repo

Dans VS Code :
- Palette de commandes (Ctrl+Shift+P ou Cmd+Shift+P)
- Cherchez "Git: Clone"
- Entrez l'URL de ce repo

### 2. Installer les dépendances

Ouvrez un terminal dans VS Code (Terminal → Nouveau Terminal) :

```bash
pip install -r requirements.txt
```

Toutes les dépendances s'installent automatiquement, y compris ffmpeg !

### 3. Configurer la clé API

**Créez un fichier `.env`** à la racine du projet :

```bash
# Sur Windows dans le terminal VS Code
type nul > .env

# Sur Mac/Linux
touch .env
```

**Ouvrez le fichier `.env`** et ajoutez votre clé :

```
OPENAI_API_KEY=sk-votre-clé-ici
```

**Sauvegardez le fichier** (Ctrl+S / Cmd+S)

## 💻 Utilisation

### Mode interactif (recommandé)

```bash
python create_subtitled_video.py
```

Le script vous demandera l'URL YouTube.

### Avec URL en paramètre

```bash
python create_subtitled_video.py "https://www.youtube.com/watch?v=VOTRE_VIDEO"
```

## 📁 Organisation des fichiers

```
Tests-Claude/
├── create_subtitled_video.py          # Script principal ⭐
├── youtube_downloader.py              # Script simple (téléchargement uniquement)
├── requirements.txt                   # Dépendances
├── .env                              # Votre clé API (à créer)
├── .env.example                      # Template
└── videos_telechargees/
    ├── videos_originales/            # Vidéos brutes téléchargées
    ├── audio_extraits/               # Fichiers audio extraits
    └── videos_sous-titrees/          # 🎉 RÉSULTATS FINAUX
```

## 🎬 Pipeline complet

Le script fait tout automatiquement :

1. **📥 Téléchargement** : Récupère la vidéo YouTube en MP4
2. **🎵 Extraction audio** : Extrait le son en MP3
3. **🎙️ Transcription** : Envoie l'audio à Whisper API → timestamps mot par mot
4. **✨ Génération** : Crée les sous-titres style TikTok (mot par mot avec surlignage)
5. **🎬 Composition** : Incruste les sous-titres dans la vidéo finale

**Temps estimé** : 5-15 minutes selon la durée de la vidéo.

## 🎨 Style des sous-titres

- Police : **Arial Bold** (grande et lisible)
- Couleur : **Blanc avec contour noir**
- Position : **Centre-haut** de la vidéo
- Animation : **Mots qui apparaissent un par un**
- Effet : **Mot actuel en MAJUSCULES** (surlignage visuel)
- Contexte : Affiche **3 mots à la fois** (précédent, actuel, suivant)

## 🐛 Dépannage

### Erreur de clé API

```
❌ Erreur : Clé API OpenAI non trouvée
```

**Solution** : Vérifiez que le fichier `.env` existe et contient `OPENAI_API_KEY=sk-...`

### Erreur SSL ou Connection Error (PC d'entreprise)

```
Connection error.
```

**Le script désactive automatiquement les problèmes suivants** :
- ✅ Vérification SSL pour les proxys d'entreprise (yt-dlp)
- ✅ Contournement des proxies pour l'API OpenAI (httpx)
- ✅ Désactivation de la vérification SSL pour OpenAI

**Si vous avez toujours une erreur** :
1. Vérifiez que vous êtes bien sur un réseau qui permet l'accès à `api.openai.com`
2. Testez avec : `python test_openai_connection.py`
3. Si le test échoue, essayez sur un autre réseau (WiFi personnel, partage de connexion)

### Erreur d'installation

```bash
# Vérifier Python
python --version

# Vérifier pip
pip --version

# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

### Vidéo trop longue

Pour les vidéos très longues (>30 min), le traitement peut prendre du temps. Soyez patient !

## 🔒 Sécurité

- ✅ Le fichier `.env` est dans `.gitignore` (ne sera JAMAIS poussé sur GitHub)
- ✅ Votre clé API reste sur votre PC local uniquement
- ✅ Aucune clé n'est visible dans les commits

## ⚠️ Notes importantes

1. **Droits d'auteur** : Respectez les conditions d'utilisation de YouTube
2. **Coût API** : Surveillez votre usage sur https://platform.openai.com/usage
3. **PC d'entreprise** : Le script fonctionne avec proxys et certificats SSL custom

## 📝 Scripts disponibles

### Test de connexion API (recommandé en premier)
```bash
python test_openai_connection.py
```
→ Teste la connexion à l'API OpenAI avant de lancer le pipeline complet

### Script complet (recommandé)
```bash
python create_subtitled_video.py
```
→ Pipeline complet : téléchargement + transcription + sous-titres

### Script simple
```bash
python youtube_downloader.py
```
→ Téléchargement uniquement (sans sous-titres)

## 🎯 Exemple d'utilisation

```bash
$ python create_subtitled_video.py

======================================================================
🎥 GÉNÉRATEUR DE SOUS-TITRES YOUTUBE STYLE TIKTOK 🎥
======================================================================
📎 URL : https://www.youtube.com/watch?v=dQw4w9WgXcQ

======================================================================
📥 ÉTAPE 1/5 : TÉLÉCHARGEMENT DE LA VIDÉO
======================================================================
✅ Vidéo téléchargée : Rick Astley - Never Gonna Give You Up

======================================================================
🎵 ÉTAPE 2/5 : EXTRACTION DE L'AUDIO
======================================================================
✅ Audio extrait : Rick Astley - Never Gonna Give You Up.mp3

======================================================================
🎙️  ÉTAPE 3/5 : TRANSCRIPTION AVEC WHISPER API
======================================================================
⏳ Envoi à l'API OpenAI... (cela peut prendre quelques instants)
✅ Transcription terminée : 487 mots détectés

======================================================================
🎬 ÉTAPE 4/5 : GÉNÉRATION DES SOUS-TITRES STYLE TIKTOK
======================================================================
⏳ Génération des sous-titres... (cela peut prendre du temps)
⏳ Composition de la vidéo finale...
⏳ Écriture du fichier vidéo... (cela peut prendre plusieurs minutes)
✅ Vidéo créée : Rick Astley - Never Gonna Give You Up_subtitled.mp4

======================================================================
✨ ÉTAPE 5/5 : TERMINÉ !
======================================================================

📁 Vidéo finale : videos_telechargees/videos_sous-titrees/Rick Astley - Never Gonna Give You Up_subtitled.mp4
📊 Taille du fichier : 25.43 MB

🎉 Processus terminé avec succès !
```

## 🚀 Améliorations futures possibles

- Choix de la couleur/police des sous-titres
- Différents styles d'animation
- Support de plus de langues
- Traitement par lots (plusieurs vidéos)
- Interface graphique

---

**Bon sous-titrage ! 🎬✨**
