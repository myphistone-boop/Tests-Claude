# 🎥 Téléchargeur de Vidéos YouTube

Un script Python simple pour télécharger des vidéos YouTube.

## 📋 Prérequis

- Python 3.6 ou supérieur
- VS Code (ou tout autre éditeur de code)

## 🚀 Installation

### 1. Cloner le repo dans VS Code

Dans VS Code, utilisez la palette de commandes (Ctrl+Shift+P) et cherchez "Git: Clone", puis entrez l'URL de ce repo.

### 2. Installer les dépendances

Ouvrez un terminal dans VS Code (Terminal → Nouveau Terminal) et exécutez :

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### Méthode 1 : Mode interactif

Lancez le script sans paramètres :

```bash
python youtube_downloader.py
```

Le script vous demandera l'URL de la vidéo à télécharger.

### Méthode 2 : Via Python

Vous pouvez aussi importer le script dans un autre fichier Python :

```python
from youtube_downloader import telecharger_video

telecharger_video("https://www.youtube.com/watch?v=VOTRE_VIDEO_ID")
```

## 📁 Organisation des fichiers

Toutes les vidéos téléchargées seront sauvegardées dans le dossier `videos_telechargees/` qui sera créé automatiquement.

## 🔧 Fonctionnalités

- ✅ Téléchargement en meilleure qualité disponible
- ✅ Affichage du titre et de la durée de la vidéo
- ✅ Barre de progression du téléchargement
- ✅ Gestion des erreurs
- ✅ Organisation automatique des fichiers

## ⚠️ Note importante

Assurez-vous de respecter les droits d'auteur et les conditions d'utilisation de YouTube lors du téléchargement de vidéos.

## 🐛 En cas de problème

Si vous rencontrez une erreur, vérifiez que :
1. Python est bien installé : `python --version`
2. Les dépendances sont installées : `pip list | grep yt-dlp`
3. Votre URL YouTube est correcte et accessible

## 📝 Exemple d'utilisation

```bash
$ python youtube_downloader.py

============================================================
🎥  TÉLÉCHARGEUR DE VIDÉOS YOUTUBE  🎥
============================================================

📎 Entrez l'URL de la vidéo YouTube : https://www.youtube.com/watch?v=dQw4w9WgXcQ

🎬 Téléchargement de la vidéo depuis : https://www.youtube.com/watch?v=dQw4w9WgXcQ
------------------------------------------------------------
📝 Titre : Rick Astley - Never Gonna Give You Up
⏱️  Durée : 3:32

[download] Destination: videos_telechargees/Rick Astley - Never Gonna Give You Up.mp4
[download] 100% of 10.5MiB in 00:05

✅ Téléchargement terminé avec succès !
📁 Vidéo sauvegardée dans : videos_telechargees/
```

## 🎯 Prochaines étapes

Ce script est une base simple. Vous pouvez l'améliorer en ajoutant :
- Choix de la qualité vidéo
- Téléchargement audio uniquement
- Téléchargement de playlists
- Interface graphique
- Et bien plus !
