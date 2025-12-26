#!/usr/bin/env python3
"""
Module de découverte et sélection de vidéos virales YouTube
Sélectionne les 10 meilleures vidéos (5 FR + 5 EN) pour la création de shorts
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import httpx
from tqdm import tqdm
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


@dataclass
class VideoCandidate:
    """Représente une vidéo candidate pour la sélection"""
    video_id: str
    title: str
    channel_title: str
    duration_seconds: int
    view_count: int
    like_count: int
    comment_count: int
    published_at: str
    language: str  # 'fr' ou 'en'
    virality_score: float = 0.0
    url: str = ""

    def __post_init__(self):
        self.url = f"https://www.youtube.com/watch?v={self.video_id}"


class YouTubeTrendingAnalyzer:
    """Analyseur de vidéos trending YouTube avec scoring de viralité"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("❌ YOUTUBE_API_KEY non trouvée dans .env")

        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.client = httpx.Client(timeout=30.0)

        # Paramètres de sélection
        self.max_duration_seconds = 15 * 60  # 15 minutes max
        self.max_results_per_language = 50  # Nombre de vidéos à analyser par langue

    def __del__(self):
        """Fermer le client HTTP"""
        if hasattr(self, 'client'):
            self.client.close()

    def get_trending_videos(self, region_code: str, language: str) -> List[VideoCandidate]:
        """
        Récupère les vidéos trending pour une région/langue donnée

        Args:
            region_code: Code région (FR, US, etc.)
            language: Code langue ('fr' ou 'en')

        Returns:
            Liste de VideoCandidate
        """
        print(f"\n🔍 Recherche des vidéos trending {language.upper()} (région: {region_code})...")

        videos = []

        # Étape 1 : Récupérer la liste des vidéos trending
        url = f"{self.base_url}/videos"
        params = {
            'part': 'snippet,contentDetails,statistics',
            'chart': 'mostPopular',
            'regionCode': region_code,
            'maxResults': self.max_results_per_language,
            'key': self.api_key
        }

        try:
            with tqdm(total=2, desc=f"   📡 API YouTube {language.upper()}", ncols=70, leave=False) as pbar:
                response = self.client.get(url, params=params)
                pbar.update(1)

                response.raise_for_status()
                data = response.json()
                pbar.update(1)

            # Traiter les résultats
            items = data.get('items', [])
            print(f"   ✅ {len(items)} vidéos récupérées")

            with tqdm(total=len(items), desc=f"   🔎 Filtrage {language.upper()}", ncols=70, leave=False) as pbar:
                for item in items:
                    try:
                        # Extraire la durée (format ISO 8601)
                        duration_iso = item['contentDetails']['duration']
                        duration_seconds = self._parse_duration(duration_iso)

                        # Filtrer par durée
                        if duration_seconds > self.max_duration_seconds:
                            pbar.update(1)
                            continue

                        # Extraire les statistiques
                        stats = item['statistics']

                        # Créer le candidat
                        video = VideoCandidate(
                            video_id=item['id'],
                            title=item['snippet']['title'],
                            channel_title=item['snippet']['channelTitle'],
                            duration_seconds=duration_seconds,
                            view_count=int(stats.get('viewCount', 0)),
                            like_count=int(stats.get('likeCount', 0)),
                            comment_count=int(stats.get('commentCount', 0)),
                            published_at=item['snippet']['publishedAt'],
                            language=language
                        )

                        videos.append(video)

                    except (KeyError, ValueError) as e:
                        # Ignorer les vidéos avec des données manquantes
                        pass

                    pbar.update(1)

            print(f"   ✅ {len(videos)} vidéos valides (≤15 min)")

        except httpx.HTTPError as e:
            print(f"   ❌ Erreur API: {e}")
            return []

        return videos

    def _parse_duration(self, duration_iso: str) -> int:
        """
        Convertit une durée ISO 8601 en secondes
        Exemple: PT1H2M10S -> 3730 secondes
        """
        import re

        # Pattern pour extraire heures, minutes, secondes
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_iso)

        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    def calculate_virality_score(self, video: VideoCandidate) -> float:
        """
        Calcule le score de viralité d'une vidéo

        Formule basée sur:
        - Ratio vues/temps depuis publication (vitesse de propagation)
        - Ratio likes/vues (engagement positif)
        - Ratio commentaires/vues (engagement conversationnel)
        - Bonus pour les vidéos récentes

        Returns:
            Score de viralité (0-100)
        """
        # Calculer l'âge de la vidéo en heures
        published = datetime.fromisoformat(video.published_at.replace('Z', '+00:00'))
        age_hours = max(1, (datetime.now(published.tzinfo) - published).total_seconds() / 3600)

        # Métriques normalisées
        views_per_hour = video.view_count / age_hours
        like_ratio = video.like_count / max(1, video.view_count)
        comment_ratio = video.comment_count / max(1, video.view_count)

        # Score pondéré
        velocity_score = min(100, views_per_hour / 1000)  # 100k vues/heure = 100 points
        engagement_score = (like_ratio * 1000) + (comment_ratio * 5000)  # Pondération
        recency_bonus = min(20, 20 / max(1, age_hours / 24))  # Bonus pour vidéos < 24h

        total_score = (velocity_score * 0.5) + (engagement_score * 0.4) + (recency_bonus * 0.1)

        return round(min(100, total_score), 2)

    def select_top_videos(self, videos: List[VideoCandidate], count: int = 5) -> List[VideoCandidate]:
        """
        Sélectionne les meilleures vidéos par score de viralité

        Args:
            videos: Liste de candidats
            count: Nombre de vidéos à sélectionner

        Returns:
            Top vidéos triées par score décroissant
        """
        print(f"\n🎯 Calcul des scores de viralité pour {len(videos)} vidéos...")

        with tqdm(total=len(videos), desc="   📊 Scoring", ncols=70, leave=False) as pbar:
            for video in videos:
                video.virality_score = self.calculate_virality_score(video)
                pbar.update(1)

        # Trier par score décroissant
        sorted_videos = sorted(videos, key=lambda v: v.virality_score, reverse=True)

        # Sélectionner le top
        top_videos = sorted_videos[:count]

        print(f"   ✅ Top {count} sélectionnées")

        return top_videos


def main():
    """Point d'entrée principal du programme de découverte"""

    print("=" * 70)
    print("🎬 DÉCOUVERTE ET SÉLECTION DE VIDÉOS VIRALES YOUTUBE")
    print("=" * 70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Objectif: Sélectionner 10 vidéos (5 FR + 5 EN)")
    print(f"⏱️  Durée max: 15 minutes")
    print("=" * 70)

    try:
        # Initialiser l'analyseur
        print("\n🔧 Initialisation...")
        analyzer = YouTubeTrendingAnalyzer()
        print("   ✅ Analyseur YouTube initialisé")

        all_selected_videos = []

        # ÉTAPE 1 : Découvrir les vidéos FR
        print("\n" + "=" * 70)
        print("ÉTAPE 1/5 : DÉCOUVERTE VIDÉOS FRANÇAISES")
        print("=" * 70)
        fr_videos = analyzer.get_trending_videos(region_code='FR', language='fr')

        # ÉTAPE 2 : Découvrir les vidéos EN
        print("\n" + "=" * 70)
        print("ÉTAPE 2/5 : DÉCOUVERTE VIDÉOS ANGLAISES")
        print("=" * 70)
        en_videos = analyzer.get_trending_videos(region_code='US', language='en')

        # ÉTAPE 3 : Calculer les scores et sélectionner FR
        print("\n" + "=" * 70)
        print("ÉTAPE 3/5 : SÉLECTION TOP 5 FRANÇAISES")
        print("=" * 70)
        top_fr = analyzer.select_top_videos(fr_videos, count=5)
        all_selected_videos.extend(top_fr)

        # ÉTAPE 4 : Calculer les scores et sélectionner EN
        print("\n" + "=" * 70)
        print("ÉTAPE 4/5 : SÉLECTION TOP 5 ANGLAISES")
        print("=" * 70)
        top_en = analyzer.select_top_videos(en_videos, count=5)
        all_selected_videos.extend(top_en)

        # ÉTAPE 5 : Sauvegarder les résultats
        print("\n" + "=" * 70)
        print("ÉTAPE 5/5 : SAUVEGARDE DES RÉSULTATS")
        print("=" * 70)

        output_dir = Path('selected_videos')
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'selected_videos_{timestamp}.json'

        # Convertir en dictionnaire
        results = {
            'timestamp': timestamp,
            'date': datetime.now().isoformat(),
            'total_selected': len(all_selected_videos),
            'french_count': len(top_fr),
            'english_count': len(top_en),
            'videos': [asdict(v) for v in all_selected_videos]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Résultats sauvegardés: {output_file}")

        # Afficher le récapitulatif
        print("\n" + "=" * 70)
        print("📊 RÉCAPITULATIF DE LA SÉLECTION")
        print("=" * 70)

        for i, video in enumerate(all_selected_videos, 1):
            lang_flag = "🇫🇷" if video.language == 'fr' else "🇺🇸"
            duration_min = video.duration_seconds // 60
            duration_sec = video.duration_seconds % 60

            print(f"\n{i}. {lang_flag} {video.title[:60]}...")
            print(f"   📺 Chaîne: {video.channel_title}")
            print(f"   ⏱️  Durée: {duration_min}m{duration_sec:02d}s")
            print(f"   👀 Vues: {video.view_count:,}")
            print(f"   👍 Likes: {video.like_count:,}")
            print(f"   💬 Commentaires: {video.comment_count:,}")
            print(f"   🔥 Score viralité: {video.virality_score}/100")
            print(f"   🔗 URL: {video.url}")

        print("\n" + "=" * 70)
        print("✅ SÉLECTION TERMINÉE AVEC SUCCÈS")
        print("=" * 70)

        return output_file

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
