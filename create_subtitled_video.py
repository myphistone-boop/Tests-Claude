#!/usr/bin/env python3
"""
Script complet pour créer des vidéos YouTube avec sous-titres style TikTok
Pipeline complet : Téléchargement -> Extraction audio -> Transcription -> Sous-titres
"""

import os
import sys
import re
import json
import random
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

    def extraire_segment_aleatoire(self, video_path, transcript, duree_souhaitee, video_title):
        """
        Extrait un segment aléatoire de la vidéo avec sa transcription

        Args:
            video_path (str): Chemin de la vidéo complète
            transcript (dict): Transcription Whisper complète
            duree_souhaitee (float): Durée souhaitée en secondes
            video_title (str): Titre de la vidéo

        Returns:
            tuple: (chemin_segment_video, transcription_segment)
        """
        print("\n" + "=" * 70)
        print("✂️  ÉTAPE 4/6 : EXTRACTION D'UN SEGMENT ALÉATOIRE")
        print("=" * 70)

        try:
            # Charger la vidéo pour obtenir sa durée
            video = VideoFileClip(video_path)
            duree_totale = video.duration

            # Vérifier que la durée souhaitée est valide
            if duree_souhaitee > duree_totale:
                print(f"⚠️  Durée souhaitée ({duree_souhaitee}s) > durée vidéo ({duree_totale:.1f}s)")
                print(f"   Utilisation de la durée maximale : {duree_totale:.1f}s")
                duree_souhaitee = duree_totale
                debut = 0
            else:
                # Choisir un point de départ aléatoire
                temps_max_debut = duree_totale - duree_souhaitee
                debut = random.uniform(0, temps_max_debut)

            fin = debut + duree_souhaitee

            print(f"📊 Durée totale de la vidéo : {duree_totale:.1f}s")
            print(f"✂️  Segment sélectionné : {debut:.1f}s → {fin:.1f}s (durée: {duree_souhaitee:.1f}s)")

            # Extraire le segment vidéo
            print("⏳ Extraction du segment vidéo...")
            segment = video.subclip(debut, fin)

            # Sauvegarder le segment
            segment_dir = self.base_dir / "segments"
            segment_dir.mkdir(parents=True, exist_ok=True)
            segment_path = segment_dir / f"{video_title}_segment_{int(debut)}_{int(fin)}.mp4"

            segment.write_videofile(
                str(segment_path),
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger='bar'
            )

            video.close()
            segment.close()

            # Filtrer la transcription pour ce segment
            print("⏳ Filtrage de la transcription pour le segment...")
            mots_filtres = []

            if hasattr(transcript, 'words') and transcript.words:
                for word in transcript.words:
                    if debut <= word.start <= fin:
                        # Ajuster les timestamps relatifs au segment
                        mots_filtres.append({
                            'word': word.word,
                            'start': word.start - debut,
                            'end': word.end - debut
                        })

            print(f"✅ Segment extrait : {len(mots_filtres)} mots dans le segment")
            print(f"✅ Fichier : {segment_path.name}")

            # Créer un objet transcription pour le segment
            class TranscriptSegment:
                def __init__(self, words_list):
                    self.words = []
                    for w in words_list:
                        word_obj = type('obj', (object,), {
                            'word': w['word'],
                            'start': w['start'],
                            'end': w['end']
                        })
                        self.words.append(word_obj)

            transcript_segment = TranscriptSegment(mots_filtres)

            return str(segment_path), transcript_segment

        except Exception as e:
            print(f"❌ Erreur lors de l'extraction du segment : {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def detecter_phrases(self, mots_timestamps):
        """
        Détecte les phrases dans la liste de mots basé sur la ponctuation

        Args:
            mots_timestamps (list): Liste des mots avec timestamps

        Returns:
            list: Liste de phrases (chaque phrase = liste de mots)
        """
        phrases = []
        phrase_courante = []

        for mot_info in mots_timestamps:
            mot = mot_info['word'].strip()
            phrase_courante.append(mot_info)

            # Détecter la fin de phrase
            if mot.endswith(('.', '!', '?', '...')) or mot in ['.', '!', '?']:
                phrases.append(phrase_courante)
                phrase_courante = []

        # Ajouter la dernière phrase si elle n'est pas vide
        if phrase_courante:
            phrases.append(phrase_courante)

        return phrases

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

        # Générer un nom de fichier unique pour éviter l'écrasement
        output_base = self.output_dir / f"{video_title}_subtitled.mp4"
        output_path = output_base

        # Si le fichier existe déjà, ajouter un numéro
        counter = 1
        while output_path.exists():
            output_path = self.output_dir / f"{video_title}_subtitled_{counter}.mp4"
            counter += 1

        if counter > 1:
            print(f"ℹ️  Fichier existant détecté, création de : {output_path.name}")

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

    def creer_video_tiktok(self, video_path, transcript, video_title):
        """
        Crée une vidéo au format TikTok (9:16) avec sous-titres cumulatifs mot par mot

        Args:
            video_path (str): Chemin de la vidéo (segment)
            transcript (dict): Transcription Whisper
            video_title (str): Titre de la vidéo

        Returns:
            str: Chemin de la vidéo finale
        """
        print("\n" + "=" * 70)
        print("📱 ÉTAPE 5/6 : CRÉATION VIDÉO FORMAT TIKTOK + SOUS-TITRES")
        print("=" * 70)

        # Générer un nom de fichier unique
        output_base = self.output_dir / f"{video_title}_tiktok.mp4"
        output_path = output_base
        counter = 1
        while output_path.exists():
            output_path = self.output_dir / f"{video_title}_tiktok_{counter}.mp4"
            counter += 1

        if counter > 1:
            print(f"ℹ️  Fichier existant détecté, création de : {output_path.name}")

        try:
            # Charger la vidéo
            print("⏳ Chargement de la vidéo...")
            video = VideoFileClip(video_path)

            # Convertir au format vertical TikTok (9:16)
            print("📱 Conversion au format vertical TikTok (9:16)...")
            target_width = 1080
            target_height = 1920

            # Redimensionner la vidéo pour qu'elle tienne dans le format 9:16
            scale = min(target_width / video.w, target_height / video.h)
            new_w = int(video.w * scale)
            new_h = int(video.h * scale)

            from moviepy.editor import ColorClip
            video_resized = video.resize((new_w, new_h))

            # Créer un fond noir
            background = ColorClip(size=(target_width, target_height), color=(0, 0, 0), duration=video.duration)

            # Centrer la vidéo sur le fond noir
            x_center = (target_width - new_w) // 2
            y_center = (target_height - new_h) // 2
            video_centered = video_resized.set_position((x_center, y_center))

            # Composer la vidéo avec le fond
            video_final = CompositeVideoClip([background, video_centered], size=(target_width, target_height))

            # Extraire les mots avec timestamps
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
                mots_timestamps = []

            # Détecter les phrases
            print("📝 Détection des phrases...")
            phrases = self.detecter_phrases(mots_timestamps)
            print(f"✅ {len(phrases)} phrases détectées")

            # Créer un dossier temporaire pour les images
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="tiktok_subs_"))
            print(f"📁 Dossier temporaire : {temp_dir}")

            # Créer les sous-titres avec accumulation
            print("⏳ Génération des sous-titres cumulatifs...")
            subtitle_clips = []

            # Charger la police une seule fois
            print("⏳ Chargement de la police...")
            try:
                font = ImageFont.truetype("arialbd.ttf", 70)
                print("✅ Police Arial Bold chargée")
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 70)
                    print("✅ Police Arial chargée")
                except:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 70)
                        print("✅ Police DejaVu chargée")
                    except:
                        font = ImageFont.load_default()
                        print("✅ Police par défaut chargée")

            # Générer les clips pour chaque phrase
            print(f"\n⏳ Traitement de {len(phrases)} phrase(s)...")
            image_counter = 0

            for phrase_idx, phrase in enumerate(phrases):
                try:
                    print(f"\n📝 Phrase {phrase_idx + 1}/{len(phrases)} - {len(phrase)} mots")

                    # Pour chaque mot de la phrase, créer un clip cumulatif
                    for mot_idx, mot_info in enumerate(phrase):
                        try:
                            # Accumuler les mots jusqu'à ce mot
                            mots_accumules = ' '.join([m['word'].strip().upper() for m in phrase[:mot_idx + 1]])

                            # Diviser en lignes si trop long
                            mots_liste = mots_accumules.split()
                            lines = []
                            current_line = []
                            for mot in mots_liste:
                                current_line.append(mot)
                                if len(' '.join(current_line)) > 15:
                                    lines.append(' '.join(current_line))
                                    current_line = []
                            if current_line:
                                lines.append(' '.join(current_line))

                            # Créer l'image de texte
                            width = target_width - 100
                            line_height = 80
                            total_height = len(lines) * line_height + 100

                            img = Image.new('RGBA', (width, total_height), (0, 0, 0, 0))
                            draw = ImageDraw.Draw(img)

                            # Dessiner chaque ligne
                            y = 20
                            for line in lines:
                                bbox = draw.textbbox((0, 0), line, font=font)
                                text_width = bbox[2] - bbox[0]
                                x = (width - text_width) // 2

                                # Contour noir
                                for offset_x in range(-3, 4):
                                    for offset_y in range(-3, 4):
                                        if offset_x != 0 or offset_y != 0:
                                            draw.text((x + offset_x, y + offset_y), line, font=font, fill='black')

                                # Texte blanc
                                draw.text((x, y), line, font=font, fill='white')
                                y += line_height

                            # Sauvegarder l'image comme fichier PNG temporaire
                            image_path = temp_dir / f"sub_{image_counter:04d}.png"
                            img.save(str(image_path))
                            image_counter += 1

                            # Créer ImageClip depuis le fichier
                            duree = mot_info['end'] - mot_info['start']
                            txt_clip = ImageClip(str(image_path), duration=duree)
                            txt_clip = txt_clip.set_start(mot_info['start'])
                            txt_clip = txt_clip.set_position(('center', target_height * 0.65))
                            subtitle_clips.append(txt_clip)

                            if (mot_idx + 1) % 5 == 0:
                                print(f"  {mot_idx + 1}/{len(phrase)} mots traités...")

                        except Exception as e:
                            print(f"\n⚠️  Erreur sur le mot '{mot_info['word']}': {e}")
                            import traceback
                            traceback.print_exc()
                            continue

                    print(f"✅ Phrase {phrase_idx + 1} terminée")

                except Exception as e:
                    print(f"\n⚠️  Erreur sur la phrase {phrase_idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"\n✅ {len(subtitle_clips)} clips de sous-titres créés ({image_counter} images générées)")

            # Composer la vidéo finale
            print("⏳ Composition de la vidéo finale...")
            final_video = CompositeVideoClip([video_final] + subtitle_clips, size=(target_width, target_height))

            # Écrire la vidéo
            print("\n" + "=" * 70)
            print("🎬 ÉTAPE 6/6 : ENCODAGE DE LA VIDÉO FINALE")
            print("=" * 70)
            print("⏳ Écriture du fichier vidéo... (cela peut prendre plusieurs minutes)")
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=30,
                verbose=False,
                logger='bar'
            )

            # Libérer les ressources
            video.close()
            video_resized.close()
            background.close()
            video_final.close()
            final_video.close()

            # Nettoyer les fichiers temporaires
            print("⏳ Nettoyage des fichiers temporaires...")
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print("✅ Fichiers temporaires supprimés")
            except Exception as e:
                print(f"⚠️  Impossible de supprimer les fichiers temporaires: {e}")

            print(f"\n✅ Vidéo TikTok créée : {output_path.name}")
            print(f"📱 Format : 1080x1920 (9:16)")
            print(f"📊 Taille du fichier : {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
            return str(output_path)

        except Exception as e:
            print(f"❌ Erreur lors de la création de la vidéo TikTok : {e}")
            import traceback
            traceback.print_exc()

            # Nettoyer les fichiers temporaires même en cas d'erreur
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass

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
        # Vérifier que stdin est disponible
        if not sys.stdin or not hasattr(sys.stdin, 'isatty'):
            print("❌ Erreur : stdin n'est pas disponible")
            print("   Le script doit être lancé dans un terminal interactif")
            sys.exit(1)

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
        print("   5. 📱 CRÉER UNE VIDÉO TIKTOK (segment aléatoire + sous-titres cumulatifs)")

        # Forcer l'affichage du prompt
        sys.stdout.flush()

        choix = input("\n👉 Votre choix (1-5) : ").strip()

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

        elif choix == "5":
            # Créer une vidéo TikTok avec segment aléatoire
            if not fichiers['videos']:
                print("❌ Aucune vidéo trouvée. Lancez l'étape 1 ou 2 d'abord.")
                sys.exit(1)

            if not fichiers['transcripts']:
                print("❌ Aucune transcription trouvée. Lancez l'étape 3 d'abord.")
                sys.exit(1)

            print("\n📹 VIDÉOS DISPONIBLES :")
            for i, video in enumerate(fichiers['videos'], 1):
                print(f"   {i}. {video.name}")

            idx_v = int(input("\n👉 Choisissez une vidéo : ").strip()) - 1
            video_path = str(fichiers['videos'][idx_v])
            video_name = fichiers['videos'][idx_v].stem

            print("\n📄 TRANSCRIPTIONS DISPONIBLES :")
            for i, transcript in enumerate(fichiers['transcripts'], 1):
                print(f"   {i}. {transcript.name}")

            idx_t = int(input("\n👉 Choisissez une transcription : ").strip()) - 1
            transcript_path = str(fichiers['transcripts'][idx_t])

            # Charger la transcription
            transcript = generator.charger_transcription(transcript_path)

            # Demander la durée souhaitée
            print("\n✂️  CONFIGURATION DU SEGMENT")
            duree_str = input("👉 Durée souhaitée du segment (en secondes, ex: 30) : ").strip()
            try:
                duree_souhaitee = float(duree_str)
                if duree_souhaitee <= 0:
                    print("❌ La durée doit être positive")
                    sys.exit(1)
            except ValueError:
                print("❌ Durée invalide")
                sys.exit(1)

            # Extraire le segment aléatoire
            segment_path, transcript_segment = generator.extraire_segment_aleatoire(
                video_path, transcript, duree_souhaitee, video_name
            )

            # Créer la vidéo TikTok
            generator.creer_video_tiktok(segment_path, transcript_segment, video_name)

            print("\n🎉 Processus terminé avec succès !")

        else:
            print("❌ Choix invalide")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Processus interrompu par l'utilisateur (Ctrl+C)")
        sys.exit(1)
    except EOFError:
        print("\n\n❌ Erreur : Impossible de lire l'entrée utilisateur")
        print("   Assurez-vous que le script est lancé dans un terminal interactif")
        print("   et non en arrière-plan ou avec stdin redirigé.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur fatale : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
