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

    def extraire_segment_fixe(self, video_path, transcript, debut, fin, video_title, mode="viral"):
        """
        Extrait un segment spécifique de la vidéo avec sa transcription

        Args:
            video_path (str): Chemin de la vidéo complète
            transcript (dict): Transcription Whisper complète
            debut (float): Timestamp de début en secondes
            fin (float): Timestamp de fin en secondes
            video_title (str): Titre de la vidéo
            mode (str): "viral" ou "aleatoire"

        Returns:
            tuple: (chemin_segment_video, transcription_segment)
        """
        mode_label = "SEGMENT VIRAL" if mode == "viral" else "SEGMENT ALÉATOIRE"
        print("\n" + "=" * 70)
        print(f"✂️  ÉTAPE 5/7 : EXTRACTION DU {mode_label}")
        print("=" * 70)

        try:
            # Charger la vidéo pour obtenir sa durée
            video = VideoFileClip(video_path)
            duree_totale = video.duration
            duree_segment = fin - debut

            print(f"📊 Durée totale de la vidéo : {duree_totale:.1f}s")
            print(f"✂️  Segment sélectionné : {debut:.1f}s → {fin:.1f}s (durée: {duree_segment:.1f}s)")

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
                    # Inclure les mots qui se chevauchent avec le segment
                    # (commence avant la fin ET se termine après le début)
                    if word.start < fin and word.end > debut:
                        # Ajuster les timestamps relatifs au segment
                        word_start = max(0, word.start - debut)
                        word_end = min(fin - debut, word.end - debut)

                        mots_filtres.append({
                            'word': word.word,
                            'start': word_start,
                            'end': word_end
                        })

            print(f"✅ Segment extrait : {len(mots_filtres)} mots dans le segment")
            print(f"✅ Fichier : {segment_path.name}")

            # Vérifier si le segment contient des mots
            if len(mots_filtres) == 0:
                print("\n⚠️  ATTENTION : Ce segment ne contient aucune parole !")
                print("   Il s'agit probablement de musique instrumentale ou de silence.")
                print("   Les sous-titres ne pourront pas être générés pour ce segment.")
                choix = input("\n👉 Voulez-vous continuer quand même ? (o/n) : ").strip().lower()
                if choix != 'o':
                    print("❌ Extraction annulée. Veuillez choisir un autre segment.")
                    sys.exit(1)

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
        # Charger la vidéo pour obtenir sa durée
        video = VideoFileClip(video_path)
        duree_totale = video.duration
        video.close()

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

        # Utiliser la méthode générique d'extraction
        return self.extraire_segment_fixe(video_path, transcript, debut, fin, video_title, mode="aleatoire")

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

    def grouper_mots(self, mots_timestamps, max_mots=3):
        """Groupe les mots par 2-3 maximum"""
        groupes = []
        groupe_actuel = []

        for mot_info in mots_timestamps:
            groupe_actuel.append(mot_info)
            # Regrouper par 2-3 mots
            if len(groupe_actuel) >= random.choice([2, 3]):
                groupes.append(groupe_actuel)
                groupe_actuel = []

        if groupe_actuel:
            groupes.append(groupe_actuel)

        return groupes

    def format_timestamp_ass(self, seconds):
        """Convertit secondes en format ASS (H:MM:SS.CC)"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)  # centisecondes
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def creer_fichier_ass_anime(self, mots_timestamps, output_path):
        """Crée un fichier ASS avec animations TikTok (blanc->jaune->blanc)"""

        # Grouper les mots par 2-3
        groupes = self.grouper_mots(mots_timestamps)

        with open(output_path, 'w', encoding='utf-8') as f:
            # En-tête ASS
            f.write("[Script Info]\n")
            f.write("Title: TikTok Animated Subtitles\n")
            f.write("ScriptType: v4.00+\n")
            f.write("WrapStyle: 0\n")
            f.write("PlayResX: 1920\n")
            f.write("PlayResY: 1080\n")
            f.write("\n")

            # Styles
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            # Style par défaut : blanc avec contour noir
            f.write("Style: Default,Arial,70,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,80,1\n")
            f.write("\n")

            # Événements
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            # Générer les sous-titres animés
            for groupe_idx, groupe in enumerate(groupes):
                start_time = groupe[0]['start']
                end_time = groupe[-1]['end']

                # Position aléatoire légèrement décalée
                pos_x = random.randint(900, 1020)  # Centre ± offset
                pos_y = random.randint(830, 920)   # Bas avec variation

                # Rotation aléatoire légère (-8° à +8°)
                rotation = random.randint(-8, 8)

                # Taille de police variable
                font_size = random.choice([65, 70, 75, 80, 85])

                # Pour chaque mot du groupe, créer un événement avec animation
                for mot_idx, mot_info in enumerate(groupe):
                    mot_start = mot_info['start']
                    mot_end = mot_info['end']

                    # Construire le texte avec animation de couleur
                    texte_groupe = ""
                    for i, m in enumerate(groupe):
                        mot_texte = m['word'].strip().upper()

                        if i == mot_idx:
                            # Mot actif = JAUNE (&H00FFFF)
                            texte_groupe += r"{\c&H00FFFF&}" + mot_texte + r"{\c&HFFFFFF&} "
                        else:
                            # Autre mot = BLANC (&HFFFFFF)
                            texte_groupe += mot_texte + " "

                    texte_groupe = texte_groupe.strip()

                    # Codes de style inline : position, rotation, taille
                    style_code = f"{{\\pos({pos_x},{pos_y})}}{{\\frz{rotation}}}{{\\fs{font_size}}}"

                    # Écrire l'événement
                    f.write(f"Dialogue: 0,{self.format_timestamp_ass(mot_start)},{self.format_timestamp_ass(mot_end)},Default,,0,0,0,,{style_code}{texte_groupe}\n")

    def analyser_moments_viraux(self, transcript, video_title, duree_cible=60, nb_segments=5):
        """
        Analyse la transcription pour identifier les moments viraux potentiels

        Args:
            transcript (dict): Transcription Whisper avec mots et timestamps
            video_title (str): Titre de la vidéo
            duree_cible (float): Durée cible des segments en secondes
            nb_segments (int): Nombre de segments à identifier (défaut: 5)

        Returns:
            dict: Analyse avec segments viraux triés par score
        """
        print("\n" + "=" * 70)
        print(f"🎯 ANALYSE IA DES MOMENTS VIRAUX (GPT-4-mini) - TOP {nb_segments}")
        print("=" * 70)

        # Vérifier si l'analyse existe déjà
        analysis_path = self.base_dir / f"{video_title}_viral_analysis_{nb_segments}seg.json"

        if analysis_path.exists():
            print(f"📂 Analyse existante trouvée : {analysis_path.name}")
            choix = input("👉 Réutiliser l'analyse existante ? (o/n) : ").strip().lower()
            if choix == 'o' or choix == '':
                with open(analysis_path, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                print(f"✅ Analyse chargée (coût API précédent : ${analysis.get('analysis_cost', 0):.4f})")
                return analysis

        # Préparer le texte complet avec timestamps
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
            print("❌ Pas de mots avec timestamps dans la transcription")
            sys.exit(1)

        # Créer le texte avec marqueurs de temps toutes les 10 secondes
        texte_avec_temps = []
        dernier_timestamp = 0
        texte_buffer = []

        for mot in mots_timestamps:
            if mot['start'] - dernier_timestamp >= 10:
                if texte_buffer:
                    texte_avec_temps.append(f"[{int(dernier_timestamp)}s] {' '.join(texte_buffer)}")
                    texte_buffer = []
                dernier_timestamp = mot['start']
            texte_buffer.append(mot['word'].strip())

        if texte_buffer:
            texte_avec_temps.append(f"[{int(dernier_timestamp)}s] {' '.join(texte_buffer)}")

        transcription_complete = '\n'.join(texte_avec_temps)

        # Prompt optimisé pour détecter les moments viraux TikTok
        prompt = f"""Analyse cette transcription de vidéo YouTube et identifie les {nb_segments} segments de {int(duree_cible)}-90 secondes avec le PLUS HAUT POTENTIEL VIRAL pour TikTok.

CRITÈRES DE VIRALITÉ TIKTOK :
1. **Hook puissant** : Phrase choc/question intrigante dans les 3 premières secondes
2. **Punchline** : Déclaration contre-intuitive, révélation surprenante
3. **Conseil actionnable** : Astuce immédiatement applicable
4. **Histoire courte** : Début/milieu/fin en 60-90 sec
5. **Émotion forte** : Surprise, colère, rire, inspiration
6. **Quotable** : Phrase mémorable et partageable
7. **Standalone** : Compréhensible sans contexte de la vidéo complète

TRANSCRIPTION :
{transcription_complete}

Réponds UNIQUEMENT avec un JSON valide (pas de markdown, pas de ```json) dans ce format exact :
{{
  "segments": [
    {{
      "rank": 1,
      "start_time": 125.0,
      "end_time": 185.0,
      "duration": 60.0,
      "viral_score": 9.5,
      "hook": "La phrase d'accroche des 3 premières secondes",
      "reason": "Pourquoi ce segment est viral (2-3 critères)",
      "category": "Conseil choc / Punchline / Histoire / Révélation"
    }}
  ]
}}

Assure-toi que les timestamps correspondent aux marqueurs [Xs] dans la transcription."""

        try:
            print("⏳ Envoi à GPT-4-mini pour analyse...")
            print(f"📊 Longueur transcription : {len(transcription_complete)} caractères")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un expert en contenu viral TikTok. Tu analyses des transcriptions pour trouver les segments avec le plus haut potentiel viral. Tu réponds UNIQUEMENT en JSON valide."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            # Calculer le coût (GPT-4-mini : $0.150/1M input, $0.600/1M output)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens / 1_000_000 * 0.150) + (output_tokens / 1_000_000 * 0.600)

            print(f"✅ Analyse terminée")
            print(f"💰 Coût API : ${cost:.4f} ({input_tokens} tokens in, {output_tokens} tokens out)")

            # Parser la réponse JSON
            response_text = response.choices[0].message.content.strip()

            # Nettoyer si l'IA a ajouté des backticks markdown
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            analysis = json.loads(response_text)
            analysis['analysis_cost'] = cost
            analysis['analyzed_at'] = str(Path(analysis_path).stat().st_mtime) if analysis_path.exists() else "now"

            # Sauvegarder l'analyse
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            print(f"💾 Analyse sauvegardée : {analysis_path.name}")

            return analysis

        except json.JSONDecodeError as e:
            print(f"❌ Erreur de parsing JSON : {e}")
            print(f"Réponse brute :\n{response_text}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse : {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def choisir_segment_viral(self, analysis, transcript, duree_souhaitee):
        """
        Affiche le menu des segments viraux et retourne le choix de l'utilisateur

        Args:
            analysis (dict): Analyse des moments viraux
            transcript (dict): Transcription complète
            duree_souhaitee (float): Durée souhaitée du segment

        Returns:
            tuple: (start_time, end_time, is_random) ou None si aléatoire choisi
        """
        segments = analysis.get('segments', [])
        nb_segments = len(segments)

        print("\n" + "=" * 70)
        print(f"🎯 TOP {nb_segments} MOMENTS VIRAUX DÉTECTÉS")
        print("=" * 70)

        if not segments:
            print("⚠️  Aucun segment viral détecté, passage en mode aléatoire")
            return None

        # Afficher les segments
        for i, segment in enumerate(segments, 1):
            print(f"\n[{i}] ⭐ {segment.get('viral_score', 0)}/10 - "
                  f"{self._format_time(segment['start_time'])} → {self._format_time(segment['end_time'])} "
                  f"({int(segment['duration'])}s)")
            print(f"    💡 Hook : \"{segment.get('hook', 'N/A')}\"")
            print(f"    📌 Raison : {segment.get('reason', 'N/A')}")
            print(f"    🏷️  Catégorie : {segment.get('category', 'N/A')}")

        print(f"\n[R] 🎲 Segment ALÉATOIRE ({int(duree_souhaitee)}s) - comme avant")
        print(f"[Q] ❌ Quitter")

        # Demander le choix
        choix = input(f"\n👉 Votre choix (1-{nb_segments}, R, Q, Entrée=R) : ").strip().upper()

        # Entrée vide = mode aléatoire par défaut
        if choix == '' or choix == 'R':
            if choix == '':
                print("ℹ️  Entrée vide détectée → Mode aléatoire activé")
            return None  # Mode aléatoire
        elif choix == 'Q':
            print("👋 Au revoir !")
            sys.exit(0)
        else:
            try:
                idx = int(choix) - 1
                if 0 <= idx < len(segments):
                    segment = segments[idx]
                    return (segment['start_time'], segment['end_time'], False)
                else:
                    print("⚠️  Choix invalide, mode aléatoire activé")
                    return None
            except ValueError:
                print("⚠️  Choix invalide, mode aléatoire activé")
                return None

    def detecter_moments_forts_local(self, transcript, duree_segment=3):
        """
        Détecte les moments forts localement sans API (gratuit)

        Args:
            transcript: Transcription Whisper avec mots
            duree_segment: Durée des moments à détecter (secondes)

        Returns:
            list: Moments forts triés par score (format: {start, end, score, reason})
        """
        if not hasattr(transcript, 'words') or not transcript.words:
            return []

        # Mots-clés viraux (score +2)
        mots_viraux = {
            'incroyable', 'jamais', 'toujours', 'secret', 'choc', 'attention',
            'important', 'urgent', 'fou', 'dingue', 'énorme', 'wow', 'omg',
            'incredible', 'never', 'always', 'secret', 'shock', 'important',
            'crazy', 'insane', 'huge', 'amazing'
        }

        # Questions (score +1.5)
        mots_questions = {'qui', 'quoi', 'pourquoi', 'comment', 'quand', 'où',
                         'who', 'what', 'why', 'how', 'when', 'where'}

        # Analyser par fenêtres glissantes
        moments = []
        mots_list = list(transcript.words)

        for i in range(len(mots_list)):
            debut = mots_list[i].start
            fin_cible = debut + duree_segment

            # Collecter les mots dans cette fenêtre
            mots_fenetre = []
            for j in range(i, len(mots_list)):
                if mots_list[j].start < fin_cible:
                    mots_fenetre.append(mots_list[j])
                else:
                    break

            if len(mots_fenetre) < 3:  # Trop court
                continue

            fin_reel = mots_fenetre[-1].end
            duree_reel = fin_reel - debut

            # Vérifier que la durée est valide (éviter division par zéro)
            if duree_reel <= 0:
                continue

            # Calculer le score
            score = 0
            raisons = []

            # 1. Densité de mots (rythme rapide = viral)
            densite = len(mots_fenetre) / duree_reel
            if densite > 3:  # Plus de 3 mots/sec
                score += 2
                raisons.append(f"Rythme rapide ({densite:.1f} mots/s)")
            elif densite > 2:
                score += 1
                raisons.append(f"Bon rythme ({densite:.1f} mots/s)")

            # 2. Mots-clés viraux
            for mot in mots_fenetre:
                mot_lower = mot.word.strip().lower()
                if mot_lower in mots_viraux:
                    score += 2
                    raisons.append(f"Mot viral: '{mot.word}'")
                    break

            # 3. Questions
            for mot in mots_fenetre:
                mot_lower = mot.word.strip().lower()
                if mot_lower in mots_questions:
                    score += 1.5
                    raisons.append("Question détectée")
                    break

            # 4. Nombres/statistiques (score +1)
            for mot in mots_fenetre:
                if any(c.isdigit() for c in mot.word):
                    score += 1
                    raisons.append(f"Statistique: '{mot.word}'")
                    break

            if score > 0:
                moments.append({
                    'start': debut,
                    'end': fin_reel,
                    'score': score,
                    'reason': ', '.join(raisons[:2]),  # Max 2 raisons
                    'words': [m.word for m in mots_fenetre]
                })

        # Trier par score décroissant
        moments.sort(key=lambda x: x['score'], reverse=True)

        return moments[:10]  # Top 10

    def creer_video_avec_hook(self, video_path, hook_start, hook_end, output_path):
        """
        Crée une vidéo avec hook de 3s au début

        Format: [Hook 3s] → [Flash transition 0.3s] → [Vidéo complète]

        Args:
            video_path: Chemin de la vidéo complète
            hook_start: Début du hook (secondes)
            hook_end: Fin du hook (secondes)
            output_path: Chemin de sortie

        Returns:
            str: Chemin de la vidéo avec hook
        """
        print("\n" + "=" * 70)
        print("🎣 AJOUT DU HOOK VIRAL AU DÉBUT")
        print("=" * 70)
        print(f"⏳ Extraction du hook ({hook_end - hook_start:.1f}s) + assemblage...")

        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip
            from moviepy.video.fx import fadein, fadeout

            # Charger la vidéo
            print("   📂 Chargement de la vidéo...")
            video = VideoFileClip(video_path)

            # Extraire le hook
            print(f"   ✂️  Extraction du hook ({self._format_time(hook_start)} → {self._format_time(hook_end)})...")
            hook = video.subclip(hook_start, hook_end)

            # Créer une transition flash blanc (0.2s)
            print("   ⚡ Création de la transition flash...")
            flash = ColorClip(size=video.size, color=(255, 255, 255), duration=0.2)
            # Ajouter l'audio de la vidéo au flash pour éviter les problèmes d'encodage
            if video.audio:
                flash = flash.set_audio(video.audio.subclip(hook_start, min(hook_start + 0.2, video.duration)))

            # Assembler: Hook → Flash → Vidéo complète
            print("   🔨 Assemblage : [Hook] → [Flash] → [Vidéo]...")
            final = concatenate_videoclips([hook, flash, video], method="compose")

            # Sauvegarder avec paramètres d'encodage compatibles
            print("\n📊 Progression de l'encodage :")
            final.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                ffmpeg_params=['-pix_fmt', 'yuv420p'],
                verbose=False,
                logger='bar'
            )

            video.close()
            final.close()

            print(f"\n✅ Hook ajouté avec succès !")
            return output_path

        except Exception as e:
            print(f"\n⚠️  Erreur lors de l'ajout du hook : {e}")
            print("   Utilisation de la vidéo sans hook...")
            return video_path

    def crop_vertical_9_16(self, video_path, output_path):
        """
        Crop la vidéo en format vertical 9:16 (TikTok/Stories)

        Args:
            video_path: Vidéo source
            output_path: Vidéo de sortie

        Returns:
            str: Chemin vidéo croppée
        """
        print("\n" + "=" * 70)
        print("📱 CONVERSION FORMAT VERTICAL 9:16")
        print("=" * 70)

        try:
            from moviepy.editor import VideoFileClip

            print("   📂 Chargement de la vidéo...")
            video = VideoFileClip(video_path)
            w, h = video.size

            print(f"   📐 Dimensions originales : {w}x{h}")

            # Calculer les dimensions 9:16
            target_ratio = 9 / 16
            current_ratio = w / h

            if current_ratio > target_ratio:
                # Vidéo trop large → crop horizontal
                # Arrondir au nombre pair (libx264 nécessite des dimensions paires)
                new_width = (int(h * target_ratio) // 2) * 2
                x_center = w // 2
                x1 = x_center - new_width // 2
                y1 = 0
                print(f"   ✂️  Crop horizontal : {new_width}x{h} (centré)")
                cropped = video.crop(x1=x1, y1=y1, width=new_width, height=h)
            else:
                # Vidéo trop haute → crop vertical
                # Arrondir au nombre pair (libx264 nécessite des dimensions paires)
                new_height = (int(w / target_ratio) // 2) * 2
                y_center = h // 2
                x1 = 0
                y1 = y_center - new_height // 2
                print(f"   ✂️  Crop vertical : {w}x{new_height} (centré)")
                cropped = video.crop(x1=x1, y1=y1, width=w, height=new_height)

            # Sauvegarder avec paramètres d'encodage compatibles
            print("\n📊 Progression de l'encodage :")
            cropped.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                ffmpeg_params=['-pix_fmt', 'yuv420p'],
                verbose=False,
                logger='bar'
            )

            video.close()
            cropped.close()

            print(f"\n✅ Format 9:16 appliqué avec succès !")
            return output_path

        except Exception as e:
            print(f"\n⚠️  Erreur lors du crop : {e}")
            print("   Utilisation de la vidéo originale...")
            return video_path

    def resize_vertical_9_16_avec_marges(self, video_path, output_path, target_height=1920):
        """
        Resize la vidéo en format vertical 9:16 AVEC MARGES (letterbox)
        Garde toute la vidéo visible en ajoutant des barres noires

        Args:
            video_path: Vidéo source
            output_path: Vidéo de sortie
            target_height: Hauteur cible (défaut 1920 pour TikTok)

        Returns:
            str: Chemin vidéo avec marges
        """
        print("\n" + "=" * 70)
        print("📱 CONVERSION FORMAT VERTICAL 9:16 (AVEC MARGES)")
        print("=" * 70)

        try:
            from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip

            print("   📂 Chargement de la vidéo...")
            video = VideoFileClip(video_path)
            w, h = video.size

            print(f"   📐 Dimensions originales : {w}x{h}")

            # Dimensions cibles 9:16
            target_ratio = 9 / 16
            target_width = (int(target_height * target_ratio) // 2) * 2

            print(f"   🎯 Dimensions cibles : {target_width}x{target_height}")

            # Calculer le resize pour que la vidéo tienne dans le cadre
            current_ratio = w / h

            if current_ratio > target_ratio:
                # Vidéo plus large → ajuster sur la largeur
                new_width = target_width
                new_height = (int(target_width / current_ratio) // 2) * 2
                resize_video = video.resize(width=new_width)
                print(f"   📏 Resize : {new_width}x{new_height} (ajusté sur largeur)")
            else:
                # Vidéo plus haute → ajuster sur la hauteur
                new_height = target_height
                new_width = (int(target_height * current_ratio) // 2) * 2
                resize_video = video.resize(height=new_height)
                print(f"   📏 Resize : {new_width}x{new_height} (ajusté sur hauteur)")

            # Créer le fond noir
            background = ColorClip(size=(target_width, target_height), color=(0, 0, 0), duration=video.duration)

            # Centrer la vidéo sur le fond
            video_centered = resize_video.set_position(('center', 'center'))

            # Composer
            print("   🎨 Ajout des marges noires...")
            final = CompositeVideoClip([background, video_centered], size=(target_width, target_height))

            # Copier l'audio
            if video.audio:
                final = final.set_audio(video.audio)

            # Sauvegarder avec paramètres d'encodage compatibles
            print("\n📊 Progression de l'encodage :")
            final.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                ffmpeg_params=['-pix_fmt', 'yuv420p'],
                verbose=False,
                logger='bar'
            )

            video.close()
            resize_video.close()
            background.close()
            final.close()

            print(f"\n✅ Format 9:16 avec marges appliqué avec succès !")
            return output_path

        except Exception as e:
            print(f"\n⚠️  Erreur lors du resize avec marges : {e}")
            print("   Utilisation de la vidéo originale...")
            return video_path

    def _format_time(self, seconds):
        """Formate les secondes en MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def creer_video_tiktok(self, video_path, transcript, video_title):
        """
        Crée une vidéo avec sous-titres mot par mot (SIMPLE)

        Args:
            video_path (str): Chemin de la vidéo (segment)
            transcript (dict): Transcription Whisper
            video_title (str): Titre de la vidéo

        Returns:
            str: Chemin de la vidéo finale
        """
        print("\n" + "=" * 70)
        print("📱 ÉTAPE 5/6 : AJOUT DES SOUS-TITRES MOT PAR MOT")
        print("=" * 70)

        # Générer un nom de fichier unique
        output_base = self.output_dir / f"{video_title}_subtitled.mp4"
        output_path = output_base
        counter = 1
        while output_path.exists():
            output_path = self.output_dir / f"{video_title}_subtitled_{counter}.mp4"
            counter += 1

        if counter > 1:
            print(f"ℹ️  Fichier existant détecté, création de : {output_path.name}")

        try:
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

            # Si pas de mots, retourner la vidéo originale sans sous-titres
            if len(mots_timestamps) == 0:
                print("⚠️  Aucun mot à sous-titrer - copie de la vidéo sans sous-titres")
                import shutil
                shutil.copy2(video_path, output_path)
                print(f"\n✅ Vidéo copiée (sans sous-titres) : {output_path.name}")
                print(f"📊 Taille du fichier : {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
                return str(output_path)

            print(f"📝 {len(mots_timestamps)} mots à sous-titrer")

            # Créer le fichier ASS avec animations TikTok
            ass_path = self.base_dir / f"{video_title}_subtitles.ass"
            print(f"⏳ Création du fichier ASS avec animations TikTok...")
            print("   🎨 Style : Groupes de 2-3 mots BLANCS → JAUNE quand parlés → BLANC")
            self.creer_fichier_ass_anime(mots_timestamps, ass_path)
            print("✅ Fichier ASS créé avec animations")

            # Utiliser ffmpeg pour incruster les sous-titres
            print("\n" + "=" * 70)
            print("🎬 INCRUSTATION SOUS-TITRES STYLE TIKTOK")
            print("=" * 70)
            print(f"⏳ Incrustation de {len(mots_timestamps)} sous-titres...")
            print("📊 Progression ffmpeg :")

            import subprocess

            # Utiliser le filtre 'ass' pour ASS avec animations inline
            cmd = [
                'ffmpeg',
                '-y',  # Écraser le fichier de sortie
                '-i', str(video_path),
                '-vf', f"ass='{str(ass_path).replace(chr(92), '/')}'",
                '-c:a', 'copy',
                '-progress', 'pipe:1',  # Afficher la progression
                '-loglevel', 'warning',  # Réduire le bruit
                str(output_path)
            ]

            # Lancer ffmpeg et afficher la progression
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

            print("   ", end='', flush=True)
            dots = 0
            for line in process.stdout:
                if 'out_time_ms' in line or 'progress' in line:
                    print(".", end='', flush=True)
                    dots += 1
                    if dots % 50 == 0:
                        print(f" [{dots} frames]")
                        print("   ", end='', flush=True)

            process.wait()

            if process.returncode != 0:
                stderr = process.stderr.read()
                print(f"\n❌ Erreur ffmpeg : {stderr}")
                sys.exit(1)

            print(f"\n✅ Incrustation terminée !")

            # Supprimer le fichier ASS
            ass_path.unlink()

            print(f"\n✅ Vidéo avec sous-titres créée : {output_path.name}")
            print(f"📊 Taille du fichier : {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
            return str(output_path)

        except Exception as e:
            print(f"❌ Erreur lors de la création de la vidéo : {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def creer_video_tiktok_optimisee(self, video_path, transcript, video_title, duree_segment=60):
        """
        Crée une vidéo TikTok OPTIMISÉE avec toutes les techniques virales

        Fonctionnalités:
        - Détection locale des moments viraux
        - Extraction d'un SEGMENT viral (60-90s)
        - Sous-titres animés TikTok
        - Hook de 3 secondes au début du segment
        - Format vertical 9:16 AVEC MARGES (garde toute la vidéo visible)

        Args:
            video_path: Vidéo source complète
            transcript: Transcription Whisper complète
            video_title: Titre
            duree_segment: Durée du segment à extraire (secondes)

        Returns:
            str: Chemin vidéo finale optimisée
        """
        print("\n" + "=" * 70)
        print("🚀 CRÉATION VIDÉO TIKTOK OPTIMISÉE")
        print("=" * 70)

        # Étape 1 : Détection locale des moments forts (GRATUIT)
        print("\n🔍 Détection locale des moments viraux...")
        moments_locaux = self.detecter_moments_forts_local(transcript, duree_segment=3)

        if moments_locaux:
            print(f"✅ {len(moments_locaux)} moments forts détectés")
            for i, moment in enumerate(moments_locaux[:5], 1):
                print(f"   {i}. {self._format_time(moment['start'])} - Score {moment['score']:.1f} ({moment['reason']})")
        else:
            print("⚠️  Aucun moment fort détecté, utilisation du début de la vidéo")

        # Étape 2 : Déterminer le segment à extraire
        if moments_locaux:
            meilleur_moment = moments_locaux[0]
            # Centrer le segment sur le moment fort
            hook_center = (meilleur_moment['start'] + meilleur_moment['end']) / 2
            segment_start = max(0, hook_center - duree_segment / 2)
            segment_end = segment_start + duree_segment

            # Vérifier qu'on ne dépasse pas la durée de la vidéo
            from moviepy.editor import VideoFileClip
            video_temp = VideoFileClip(video_path)
            video_duration = video_temp.duration
            video_temp.close()

            if segment_end > video_duration:
                segment_end = video_duration
                segment_start = max(0, video_duration - duree_segment)

            print(f"\n📍 Segment viral sélectionné : {self._format_time(segment_start)} → {self._format_time(segment_end)}")
            print(f"   Centré sur le meilleur moment ({meilleur_moment['reason']})")

            # Position du hook DANS le segment
            hook_start_in_segment = meilleur_moment['start'] - segment_start
            hook_end_in_segment = meilleur_moment['end'] - segment_start
        else:
            # Pas de moment fort, prendre le début
            segment_start = 0
            segment_end = min(duree_segment, video_duration)
            hook_start_in_segment = 0
            hook_end_in_segment = 3

        # Étape 3 : Extraire le segment viral
        print(f"\n✂️  Extraction du segment ({segment_end - segment_start:.0f}s)...")
        segment_path, transcript_segment = self.extraire_segment_fixe(
            video_path,
            transcript,
            segment_start,
            segment_end,
            video_title,
            mode="viral"
        )

        # Étape 4 : Ajouter les sous-titres sur le segment
        print("\n📝 Ajout des sous-titres sur le segment...")
        video_subtitled = self.creer_video_tiktok(segment_path, transcript_segment, f"{video_title}_temp")

        # Étape 5 : Ajouter le hook au début du segment
        print(f"\n🎣 Hook : {self._format_time(hook_start_in_segment)} → {self._format_time(hook_end_in_segment)} du segment")
        video_hook_path = self.output_dir / f"{video_title}_with_hook.mp4"
        video_with_hook = self.creer_video_avec_hook(
            video_subtitled,
            hook_start_in_segment,
            hook_end_in_segment,
            str(video_hook_path)
        )

        # Étape 6 : Format vertical 9:16 avec marges (garde toute la vidéo visible)
        final_path = self.output_dir / f"{video_title}_optimized.mp4"
        video_final = self.resize_vertical_9_16_avec_marges(video_with_hook, str(final_path))

        print("\n" + "=" * 70)
        print("✨ VIDÉO OPTIMISÉE TERMINÉE !")
        print("=" * 70)
        print(f"📁 Fichier : {Path(video_final).name}")
        print(f"📊 Taille : {Path(video_final).stat().st_size / (1024*1024):.2f} MB")
        print(f"⏱️  Durée : {segment_end - segment_start:.0f}s")
        print("\n🎉 Prête pour TikTok/YouTube Shorts !")

        return video_final

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

            # Si on vient de la partie 1, continuer automatiquement vers Short TikTok complet
            if etape_depart == 1:
                print("\n" + "=" * 70)
                print("🚀 CRÉATION AUTOMATIQUE DU SHORT TIKTOK")
                print("=" * 70)
                print("➡️  Étape suivante : Création du short TikTok complet (1min + sous-titres + portrait)")

                # Continuer automatiquement vers Option 6
                return self.creer_video_tiktok_optimisee(video_path, transcript, video_title)

            # Pour les autres étapes de départ (2-3), demander confirmation
            elif etape_depart <= 3:  # Seulement si on vient de faire la transcription
                print("\n" + "=" * 70)
                print("🎬 ÉTAPE SUIVANTE : GÉNÉRATION DES SOUS-TITRES")
                print("=" * 70)
                print("Vous pouvez maintenant :")
                print("   1. Générer les sous-titres pour la vidéo complète (maintenant)")
                print("   2. Arrêter ici et générer les sous-titres plus tard (option 4 ou 5)")

                choix_suite = input("\n👉 Voulez-vous générer les sous-titres maintenant ? (o/n) : ").strip().lower()

                if choix_suite != 'o':
                    print("\n✅ Transcription terminée et sauvegardée !")
                    print(f"📁 Vidéo : {video_path}")
                    print(f"📁 Audio : {audio_path}")
                    print(f"📁 Transcription : {self.transcripts_dir / f'{video_title}_transcript.json'}")
                    print("\n💡 Pour générer les sous-titres plus tard, relancez le script et choisissez :")
                    print("   - Option 4 : Sous-titrer la vidéo complète")
                    print("   - Option 5 : Créer un segment TikTok viral")
                    return  # Arrêter ici

        # Étape 4 : Charger la transcription si on démarre ici
        if etape_depart == 4:
            if not video_title:
                video_title = Path(transcript_path).stem.replace('_transcript', '')
            print("\n" + "=" * 70)
            print("📂 CHARGEMENT DE LA TRANSCRIPTION EXISTANTE")
            print("=" * 70)
            transcript = self.charger_transcription(transcript_path)
            print(f"✅ Transcription chargée : {Path(transcript_path).name}")

        # Étape 4 : Créer la vidéo avec sous-titres (méthode rapide)
        output_path = self.creer_video_tiktok(video_path, transcript, video_title)

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
        print("   1. 🚀 TÉLÉCHARGER & CRÉER SHORT TIKTOK (pipeline complet 1→2→3→6)")
        print("   2. Utiliser une vidéo existante (extraire l'audio)")
        print("   3. Utiliser un audio existant (transcrire)")
        print("   4. ⚡ Sous-titrer la VIDÉO COMPLÈTE (méthode rapide ffmpeg/ASS)")
        print("   5. 📱 CRÉER UNE VIDÉO TIKTOK (segment viral + analyse IA)")
        print("   6. 🚀 SHORT TIKTOK COMPLET (1min + sous-titres + portrait 9:16 avec marges)")
        print("   7. 📱 CONVERTIR EN FORMAT VERTICAL 9:16 (crop sans marges)")

        # Forcer l'affichage du prompt
        sys.stdout.flush()

        choix = input("\n👉 Votre choix (1-7) : ").strip()

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
            # Sous-titrer la vidéo complète avec la méthode rapide (ffmpeg/ASS)
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

            # Charger la transcription
            print("\n" + "=" * 70)
            print("📂 CHARGEMENT DE LA TRANSCRIPTION")
            print("=" * 70)
            transcript = generator.charger_transcription(transcript_path)
            print(f"✅ Transcription chargée : {Path(transcript_path).name}")

            # Sous-titrer la vidéo complète avec la méthode rapide
            video_name = Path(video_path).stem
            generator.creer_video_tiktok(video_path, transcript, video_name)

            print("\n🎉 Processus terminé avec succès !")

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
            duree_str = input("👉 Durée souhaitée du segment (en secondes, ex: 60) : ").strip()
            try:
                duree_souhaitee = float(duree_str)
                if duree_souhaitee <= 0:
                    print("❌ La durée doit être positive")
                    sys.exit(1)
            except ValueError:
                print("❌ Durée invalide")
                sys.exit(1)

            # Demander le nombre de segments à analyser
            print("\n💰 OPTIMISATION DES COÛTS API")
            print("   Plus de segments = meilleure sélection, mais coût API plus élevé")
            print("   Recommandation : 3-5 segments pour un bon compromis")
            nb_segments_str = input("👉 Nombre de segments viraux à identifier (1-10, défaut: 5) : ").strip()
            try:
                if nb_segments_str == "":
                    nb_segments = 5
                else:
                    nb_segments = int(nb_segments_str)
                    if nb_segments < 1 or nb_segments > 10:
                        print("⚠️  Nombre invalide, utilisation de la valeur par défaut (5)")
                        nb_segments = 5
            except ValueError:
                print("⚠️  Nombre invalide, utilisation de la valeur par défaut (5)")
                nb_segments = 5

            # PHASE 4 : Analyser les moments viraux avec GPT-4-mini
            analysis = generator.analyser_moments_viraux(transcript, video_name, duree_cible=duree_souhaitee, nb_segments=nb_segments)

            # PHASE 4.5 : Choisir le segment (viral ou aléatoire)
            choix_segment = generator.choisir_segment_viral(analysis, transcript, duree_souhaitee)

            # PHASE 5 : Extraire le segment choisi
            if choix_segment is None:
                # Mode aléatoire
                print("\n🎲 Mode aléatoire sélectionné")
                segment_path, transcript_segment = generator.extraire_segment_aleatoire(
                    video_path, transcript, duree_souhaitee, video_name
                )
            else:
                # Mode viral - segment spécifique
                debut, fin, _ = choix_segment
                print(f"\n⭐ Segment viral sélectionné : {generator._format_time(debut)} → {generator._format_time(fin)}")
                segment_path, transcript_segment = generator.extraire_segment_fixe(
                    video_path, transcript, debut, fin, video_name, mode="viral"
                )

            # PHASE 6 : Créer la vidéo TikTok avec sous-titres animés
            generator.creer_video_tiktok(segment_path, transcript_segment, video_name)

            print("\n🎉 Processus terminé avec succès !")

        elif choix == "6":
            # Créer une vidéo TikTok OPTIMISÉE (hook + détection locale + vertical)
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

            # Créer la vidéo optimisée
            generator.creer_video_tiktok_optimisee(video_path, transcript, video_name)

            print("\n🎉 Processus terminé avec succès !")

        elif choix == "7":
            # Convertir une vidéo en format vertical 9:16
            if not fichiers['videos']:
                print("❌ Aucune vidéo trouvée. Lancez l'étape 1 ou 2 d'abord.")
                sys.exit(1)

            print("\n📹 VIDÉOS DISPONIBLES :")
            for i, video in enumerate(fichiers['videos'], 1):
                print(f"   {i}. {video.name}")

            idx_v = int(input("\n👉 Choisissez une vidéo : ").strip()) - 1
            video_path = str(fichiers['videos'][idx_v])
            video_name = fichiers['videos'][idx_v].stem

            # Convertir en format vertical
            output_path = generator.output_dir / f"{video_name}_vertical.mp4"
            generator.crop_vertical_9_16(video_path, str(output_path))

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
