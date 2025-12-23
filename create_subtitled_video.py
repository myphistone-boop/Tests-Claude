#!/usr/bin/env python3
"""
Script complet pour créer des vidéos YouTube avec sous-titres style TikTok
Pipeline complet : Téléchargement -> Extraction audio -> Transcription -> Sous-titres
"""

import os
import sys
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import yt_dlp
from openai import OpenAI
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

# Charger les variables d'environnement
load_dotenv()


def nettoyer_nom_fichier(nom):
    """Nettoie un nom de fichier en supprimant les caractères problématiques"""
    # Supprimer les caractères invalides pour les noms de fichiers
    nom = re.sub(r'[<>:"/\\|?*@]', '', nom)
    # Supprimer les guillemets doubles
    nom = nom.replace('"', '').replace("'", '')
    # Limiter la longueur à 200 caractères
    if len(nom) > 200:
        nom = nom[:200]
    return nom.strip()


class YouTubeSubtitleGenerator:
    """Classe pour gérer tout le pipeline de génération de sous-titres"""

    def __init__(self):
        # Vérifier la clé API OpenAI
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ Erreur : Clé API OpenAI non trouvée.\n"
                "Créez un fichier .env avec : OPENAI_API_KEY=votre-clé"
            )

        # Désactiver le proxy pour OpenAI (nécessaire sur certains environnements)
        import httpx
        http_client = httpx.Client(
            trust_env=False,  # Ignore les variables d'environnement proxy
            verify=False,     # Désactive la vérification SSL si nécessaire
            timeout=60.0
        )

        self.client = OpenAI(
            api_key=self.api_key,
            http_client=http_client
        )

        # Créer les dossiers de sortie
        self.base_dir = Path("videos_telechargees")
        self.videos_dir = self.base_dir / "videos_originales"
        self.audio_dir = self.base_dir / "audio_extraits"
        self.output_dir = self.base_dir / "videos_sous-titrees"

        self.transcripts_dir = self.base_dir / "transcriptions"

        for directory in [self.videos_dir, self.audio_dir, self.output_dir, self.transcripts_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def detecter_fichiers_existants(self, url=None):
        """Détecte les fichiers existants dans les dossiers"""
        videos = list(self.videos_dir.glob("*.mp4")) + list(self.videos_dir.glob("*.webm"))
        audios = list(self.audio_dir.glob("*.mp3"))
        transcripts = list(self.transcripts_dir.glob("*.json"))

        return {
            'videos': videos,
            'audios': audios,
            'transcripts': transcripts
        }

    def sauvegarder_transcription(self, transcript, video_title):
        """Sauvegarde la transcription en JSON"""
        transcript_path = self.transcripts_dir / f"{video_title}_transcript.json"

        # Convertir l'objet Whisper en dict
        data = {
            'text': transcript.text if hasattr(transcript, 'text') else '',
            'words': []
        }

        if hasattr(transcript, 'words') and transcript.words:
            data['words'] = [
                {
                    'word': word.word,
                    'start': word.start,
                    'end': word.end
                }
                for word in transcript.words
            ]
        elif hasattr(transcript, 'segments') and transcript.segments:
            data['segments'] = [
                {
                    'text': seg.text,
                    'start': seg.start,
                    'end': seg.end
                }
                for seg in transcript.segments
            ]

        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 Transcription sauvegardée : {transcript_path.name}")
        return str(transcript_path)

    def charger_transcription(self, transcript_path):
        """Charge une transcription depuis JSON"""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Créer un objet similaire à celui de Whisper
        class TranscriptData:
            def __init__(self, data):
                self.text = data.get('text', '')
                self.words = []
                self.segments = []

                if 'words' in data:
                    for w in data['words']:
                        word_obj = type('Word', (), w)()
                        self.words.append(word_obj)

                if 'segments' in data:
                    for s in data['segments']:
                        seg_obj = type('Segment', (), s)()
                        self.segments.append(seg_obj)

        return TranscriptData(data)

    def telecharger_video(self, url):
        """
        Télécharge une vidéo YouTube

        Args:
            url (str): URL de la vidéo YouTube

        Returns:
            str: Chemin du fichier vidéo téléchargé
        """
        print("\n" + "=" * 70)
        print("📥 ÉTAPE 1/5 : TÉLÉCHARGEMENT DE LA VIDÉO")
        print("=" * 70)

        try:
            # Première passe : récupérer les infos sans télécharger
            print("⏳ Récupération des informations de la vidéo...")
            with yt_dlp.YoutubeDL({'quiet': True, 'nocheckcertificate': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title_original = info.get('title', 'video')
                video_title_clean = nettoyer_nom_fichier(video_title_original)
                video_ext = info.get('ext', 'mp4')

            print(f"📹 Titre : {video_title_clean}")
            print("⏳ Téléchargement en cours...")

            # Télécharger avec le nom nettoyé directement
            video_path_clean = self.videos_dir / f"{video_title_clean}.{video_ext}"

            options = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': str(video_path_clean),
                'quiet': False,
                'no_warnings': False,
                'nocheckcertificate': True,
            }

            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])

            print(f"\n✅ Vidéo téléchargée : {video_title_clean}")
            return str(video_path_clean), video_title_clean

        except Exception as e:
            print(f"❌ Erreur lors du téléchargement : {e}")
            sys.exit(1)

    def extraire_audio(self, video_path, video_title):
        """
        Extrait l'audio de la vidéo

        Args:
            video_path (str): Chemin de la vidéo
            video_title (str): Titre de la vidéo

        Returns:
            str: Chemin du fichier audio
        """
        print("\n" + "=" * 70)
        print("🎵 ÉTAPE 2/5 : EXTRACTION DE L'AUDIO")
        print("=" * 70)

        audio_path = self.audio_dir / f"{video_title}.mp3"

        try:
            print("⏳ Chargement de la vidéo...")
            video = VideoFileClip(video_path)

            print(f"⏳ Extraction de l'audio ({video.duration:.1f}s)...")
            video.audio.write_audiofile(
                str(audio_path),
                codec='mp3',
                verbose=False,
                logger='bar'  # Affiche une barre de progression
            )
            video.close()

            print(f"✅ Audio extrait : {audio_path.name}")
            return str(audio_path)

        except Exception as e:
            print(f"❌ Erreur lors de l'extraction audio : {e}")
            sys.exit(1)

    def transcrire_avec_whisper(self, audio_path, video_title):
        """
        Transcrit l'audio avec l'API Whisper d'OpenAI

        Args:
            audio_path (str): Chemin du fichier audio
            video_title (str): Titre de la vidéo (pour sauvegarder)

        Returns:
            dict: Transcription avec timestamps mot par mot
        """
        print("\n" + "=" * 70)
        print("🎙️  ÉTAPE 3/5 : TRANSCRIPTION AVEC WHISPER API")
        print("=" * 70)

        # Calculer la durée de l'audio pour estimer le coût
        try:
            from moviepy.editor import AudioFileClip
            audio_clip = AudioFileClip(audio_path)
            duree_secondes = audio_clip.duration
            audio_clip.close()
            duree_minutes = duree_secondes / 60

            print(f"📊 Durée de l'audio : {duree_minutes:.2f} minutes ({duree_secondes:.0f}s)")
            print(f"💰 Coût estimé : ${duree_minutes * 0.006:.4f} USD")
        except Exception as e:
            print(f"⚠️  Impossible de calculer la durée : {e}")
            duree_minutes = 0

        print("⏳ Envoi à l'API OpenAI... (cela peut prendre quelques instants)")

        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            # Vérifier si on a des timestamps de mots
            if hasattr(transcript, 'words') and transcript.words:
                nb_mots = len(transcript.words)
                print(f"✅ Transcription terminée : {nb_mots} mots détectés")
            else:
                print("⚠️  Attention : Pas de timestamps mot par mot disponibles")
                print("   Utilisation des segments à la place...")

            # Afficher le coût réel
            if duree_minutes > 0:
                cout_reel = duree_minutes * 0.006
                print(f"\n💵 COÛT DE L'APPEL API WHISPER : ${cout_reel:.4f} USD")
                print(f"   (Tarif : $0.006/minute × {duree_minutes:.2f} minutes)")

            # Sauvegarder la transcription
            self.sauvegarder_transcription(transcript, video_title)

            return transcript

        except Exception as e:
            print(f"❌ Erreur lors de la transcription : {e}")
            sys.exit(1)

    def creer_fonction_sous_titres(self, transcript):
        """
        Crée une fonction de génération de sous-titres pour moviepy

        Args:
            transcript (dict): Transcription Whisper

        Returns:
            function: Fonction qui retourne le texte à afficher à un instant t
        """
        # Extraire les mots avec leurs timestamps
        if hasattr(transcript, 'words') and transcript.words:
            mots_timestamps = [
                {
                    'word': word.word,
                    'start': word.start,
                    'end': word.end
                }
                for word in transcript.words
            ]
        else:
            # Fallback sur les segments si pas de mots
            mots_timestamps = []
            for segment in transcript.segments:
                mots_timestamps.append({
                    'word': segment.text,
                    'start': segment.start,
                    'end': segment.end
                })

        def generer_texte(t):
            """Retourne le texte à afficher au temps t - UN SEUL MOT"""
            # Trouver le mot actuel
            for mot_info in mots_timestamps:
                if mot_info['start'] <= t <= mot_info['end']:
                    # Retourner uniquement le mot actuel en MAJUSCULES
                    return mot_info['word'].strip().upper()

            return ""

        return generer_texte, mots_timestamps

    def creer_video_sous_titree(self, video_path, transcript, video_title):
        """
        Crée la vidéo finale avec sous-titres incrustés

        Args:
            video_path (str): Chemin de la vidéo originale
            transcript (dict): Transcription Whisper
            video_title (str): Titre de la vidéo

        Returns:
            str: Chemin de la vidéo finale
        """
        print("\n" + "=" * 70)
        print("🎬 ÉTAPE 4/5 : GÉNÉRATION DES SOUS-TITRES STYLE TIKTOK")
        print("=" * 70)

        output_path = self.output_dir / f"{video_title}_subtitled.mp4"

        try:
            # Charger la vidéo
            video = VideoFileClip(video_path)

            # Créer la fonction de sous-titres
            generer_texte, mots_timestamps = self.creer_fonction_sous_titres(transcript)

            # Créer les clips de texte
            duree_totale = mots_timestamps[-1]['end'] if mots_timestamps else video.duration

            print("⏳ Génération des sous-titres... (cela peut prendre du temps)")

            # Créer un clip de sous-titres avec PIL (pas besoin d'ImageMagick)
            def make_textclip(txt):
                if not txt:
                    return None

                # Dimensions de l'image
                width = int(video.w * 0.9)
                height = 200  # Hauteur suffisante pour le texte

                # Créer une image transparente
                img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                # Essayer de charger une police, sinon utiliser la police par défaut
                try:
                    # Essayer Arial Bold
                    font = ImageFont.truetype("arialbd.ttf", 60)
                except:
                    try:
                        # Essayer Arial normale
                        font = ImageFont.truetype("arial.ttf", 60)
                    except:
                        try:
                            # Pour Linux
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
                        except:
                            # Police par défaut
                            font = ImageFont.load_default()

                # Calculer la position du texte pour le centrer
                bbox = draw.textbbox((0, 0), txt, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (width - text_width) // 2
                y = (height - text_height) // 2

                # Dessiner le contour noir (stroke)
                stroke_width = 3
                for offset_x in range(-stroke_width, stroke_width + 1):
                    for offset_y in range(-stroke_width, stroke_width + 1):
                        draw.text((x + offset_x, y + offset_y), txt, font=font, fill='black')

                # Dessiner le texte blanc par-dessus
                draw.text((x, y), txt, font=font, fill='white')

                # Convertir en array numpy pour moviepy
                img_array = np.array(img)

                # Créer un ImageClip
                return ImageClip(img_array, duration=0.1)

            # Créer les sous-titres pour chaque mot
            subtitle_clips = []

            print(f"📝 Génération de {len(mots_timestamps)} sous-titres...")
            for mot_info in tqdm(mots_timestamps, desc="Sous-titres", unit="mot"):
                start = mot_info['start']
                end = mot_info['end']

                # Récupérer le mot actuel
                txt = generer_texte((start + end) / 2)

                if txt:
                    txt_clip = make_textclip(txt)
                    if txt_clip:
                        txt_clip = txt_clip.set_start(start).set_end(end)
                        txt_clip = txt_clip.set_position(('center', video.h * 0.15))
                        subtitle_clips.append(txt_clip)

            # Composer la vidéo finale
            print("⏳ Composition de la vidéo finale...")
            final_video = CompositeVideoClip([video] + subtitle_clips)

            # Écrire la vidéo
            print("\n" + "=" * 70)
            print("🎬 ÉTAPE 5/5 : COMPOSITION DE LA VIDÉO FINALE")
            print("=" * 70)
            print("⏳ Écriture du fichier vidéo... (cela peut prendre plusieurs minutes)")
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger='bar'  # Affiche une barre de progression
            )

            # Libérer les ressources
            video.close()
            final_video.close()

            print(f"✅ Vidéo créée : {output_path.name}")
            return str(output_path)

        except Exception as e:
            print(f"❌ Erreur lors de la création de la vidéo : {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def traiter_video(self, url=None, etape_depart=1, video_path=None, audio_path=None, transcript_path=None, video_title=None):
        """
        Pipeline complet ou partiel : téléchargement -> transcription -> sous-titres

        Args:
            url (str): URL YouTube (si etape_depart=1)
            etape_depart (int): Étape de départ (1-4)
            video_path (str): Chemin vidéo existante (si etape_depart>=2)
            audio_path (str): Chemin audio existant (si etape_depart>=3)
            transcript_path (str): Chemin transcription existante (si etape_depart>=4)
            video_title (str): Titre de la vidéo (optionnel)
        """
        print("\n" + "=" * 70)
        print("🎥 GÉNÉRATEUR DE SOUS-TITRES YOUTUBE STYLE TIKTOK 🎥")
        print("=" * 70)

        transcript = None

        # Étape 1 : Télécharger la vidéo
        if etape_depart <= 1:
            print(f"📎 URL : {url}\n")
            video_path, video_title = self.telecharger_video(url)

        # Étape 2 : Extraire l'audio
        if etape_depart <= 2:
            if not video_title:
                video_title = Path(video_path).stem
            audio_path = self.extraire_audio(video_path, video_title)

        # Étape 3 : Transcrire avec Whisper
        if etape_depart <= 3:
            if not video_title:
                video_title = Path(audio_path).stem
            transcript = self.transcrire_avec_whisper(audio_path, video_title)

        # Étape 4 : Charger la transcription si on démarre ici
        if etape_depart == 4:
            if not video_title:
                video_title = Path(transcript_path).stem.replace('_transcript', '')
            print("\n" + "=" * 70)
            print("📂 CHARGEMENT DE LA TRANSCRIPTION EXISTANTE")
            print("=" * 70)
            transcript = self.charger_transcription(transcript_path)
            print(f"✅ Transcription chargée : {Path(transcript_path).name}")

        # Étape 4 : Créer la vidéo avec sous-titres
        output_path = self.creer_video_sous_titree(video_path, transcript, video_title)

        # Étape 5 : Résumé final
        print("\n" + "=" * 70)
        print("✨ ÉTAPE 5/5 : TERMINÉ !")
        print("=" * 70)
        print(f"\n📁 Vidéo finale : {output_path}")
        print(f"📊 Taille du fichier : {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
        print("\n🎉 Processus terminé avec succès !")


def main():
    """Fonction principale"""
    try:
        generator = YouTubeSubtitleGenerator()

        print("\n" + "=" * 70)
        print("🎥 GÉNÉRATEUR DE SOUS-TITRES YOUTUBE STYLE TIKTOK 🎥")
        print("=" * 70)

        # Détecter les fichiers existants
        fichiers = generator.detecter_fichiers_existants()

        print("\n📂 DÉTECTION DES FICHIERS EXISTANTS :")
        print(f"   Vidéos : {len(fichiers['videos'])} fichier(s)")
        print(f"   Audios : {len(fichiers['audios'])} fichier(s)")
        print(f"   Transcriptions : {len(fichiers['transcripts'])} fichier(s)")

        print("\n🎬 CHOISISSEZ L'ÉTAPE DE DÉPART :")
        print("   1. Télécharger une nouvelle vidéo YouTube (tout recommencer)")
        print("   2. Utiliser une vidéo existante (extraire l'audio)")
        print("   3. Utiliser un audio existant (transcrire)")
        print("   4. Utiliser une transcription existante (générer sous-titres)")

        choix = input("\n👉 Votre choix (1-4) : ").strip()

        if choix == "1":
            # Nouveau téléchargement
            url = input("\n📎 Entrez l'URL de la vidéo YouTube : ").strip()
            if not url:
                print("❌ Erreur : URL vide")
                sys.exit(1)
            generator.traiter_video(url=url, etape_depart=1)

        elif choix == "2":
            # Utiliser vidéo existante
            if not fichiers['videos']:
                print("❌ Aucune vidéo trouvée. Lancez l'étape 1 d'abord.")
                sys.exit(1)

            print("\n📹 VIDÉOS DISPONIBLES :")
            for i, video in enumerate(fichiers['videos'], 1):
                print(f"   {i}. {video.name}")

            idx = int(input("\n👉 Choisissez une vidéo : ").strip()) - 1
            video_path = str(fichiers['videos'][idx])
            generator.traiter_video(etape_depart=2, video_path=video_path)

        elif choix == "3":
            # Utiliser audio existant
            if not fichiers['audios']:
                print("❌ Aucun audio trouvé. Lancez l'étape 1 ou 2 d'abord.")
                sys.exit(1)

            if not fichiers['videos']:
                print("❌ Aucune vidéo trouvée. La vidéo est nécessaire pour la génération finale.")
                sys.exit(1)

            print("\n🎵 AUDIOS DISPONIBLES :")
            for i, audio in enumerate(fichiers['audios'], 1):
                print(f"   {i}. {audio.name}")

            idx = int(input("\n👉 Choisissez un audio : ").strip()) - 1
            audio_path = str(fichiers['audios'][idx])

            # Trouver la vidéo correspondante
            audio_name = fichiers['audios'][idx].stem
            video_path = None
            for v in fichiers['videos']:
                if v.stem == audio_name:
                    video_path = str(v)
                    break

            if not video_path:
                print(f"⚠️  Vidéo correspondante non trouvée. Veuillez sélectionner une vidéo :")
                for i, video in enumerate(fichiers['videos'], 1):
                    print(f"   {i}. {video.name}")
                idx_v = int(input("\n👉 Choisissez une vidéo : ").strip()) - 1
                video_path = str(fichiers['videos'][idx_v])

            generator.traiter_video(etape_depart=3, video_path=video_path, audio_path=audio_path)

        elif choix == "4":
            # Utiliser transcription existante
            if not fichiers['transcripts']:
                print("❌ Aucune transcription trouvée. Lancez l'étape 1, 2 ou 3 d'abord.")
                sys.exit(1)

            if not fichiers['videos']:
                print("❌ Aucune vidéo trouvée. La vidéo est nécessaire pour la génération finale.")
                sys.exit(1)

            print("\n📄 TRANSCRIPTIONS DISPONIBLES :")
            for i, transcript in enumerate(fichiers['transcripts'], 1):
                print(f"   {i}. {transcript.name}")

            idx = int(input("\n👉 Choisissez une transcription : ").strip()) - 1
            transcript_path = str(fichiers['transcripts'][idx])

            # Trouver la vidéo correspondante
            transcript_name = fichiers['transcripts'][idx].stem.replace('_transcript', '')
            video_path = None
            for v in fichiers['videos']:
                if v.stem == transcript_name:
                    video_path = str(v)
                    break

            if not video_path:
                print(f"⚠️  Vidéo correspondante non trouvée. Veuillez sélectionner une vidéo :")
                for i, video in enumerate(fichiers['videos'], 1):
                    print(f"   {i}. {video.name}")
                idx_v = int(input("\n👉 Choisissez une vidéo : ").strip()) - 1
                video_path = str(fichiers['videos'][idx_v])

            generator.traiter_video(etape_depart=4, video_path=video_path, transcript_path=transcript_path)

        else:
            print("❌ Choix invalide")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Processus interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur fatale : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
