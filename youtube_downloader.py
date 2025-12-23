#!/usr/bin/env python3
"""
Script simple pour télécharger des vidéos YouTube
"""

import yt_dlp
import os


def telecharger_video(url):
    """
    Télécharge une vidéo YouTube à partir de son URL

    Args:
        url (str): L'URL de la vidéo YouTube
    """
    # Créer un dossier pour les téléchargements s'il n'existe pas
    dossier_telechargement = "videos_telechargees"
    if not os.path.exists(dossier_telechargement):
        os.makedirs(dossier_telechargement)

    # Configuration des options de téléchargement
    options = {
        'format': 'best',  # Meilleure qualité disponible
        'outtmpl': f'{dossier_telechargement}/%(title)s.%(ext)s',  # Nom du fichier
        'quiet': False,  # Afficher la progression
        'no_warnings': False,
        'nocheckcertificate': True,  # Désactiver vérification SSL (nécessaire sur PC d'entreprise avec proxy)
    }

    try:
        print(f"\n🎬 Téléchargement de la vidéo depuis : {url}")
        print("-" * 60)

        with yt_dlp.YoutubeDL(options) as ydl:
            # Récupérer les informations de la vidéo
            info = ydl.extract_info(url, download=False)
            titre = info.get('title', 'Titre inconnu')
            duree = info.get('duration', 0)

            print(f"📝 Titre : {titre}")
            print(f"⏱️  Durée : {duree // 60}:{duree % 60:02d}")
            print()

            # Télécharger la vidéo
            ydl.download([url])

        print("\n✅ Téléchargement terminé avec succès !")
        print(f"📁 Vidéo sauvegardée dans : {dossier_telechargement}/")

    except Exception as e:
        print(f"\n❌ Erreur lors du téléchargement : {e}")


def main():
    """Fonction principale"""
    print("=" * 60)
    print("🎥  TÉLÉCHARGEUR DE VIDÉOS YOUTUBE  🎥")
    print("=" * 60)

    # Demander l'URL à l'utilisateur
    url = input("\n📎 Entrez l'URL de la vidéo YouTube : ").strip()

    if not url:
        print("❌ Erreur : URL vide. Veuillez entrer une URL valide.")
        return

    # Télécharger la vidéo
    telecharger_video(url)


if __name__ == "__main__":
    main()
