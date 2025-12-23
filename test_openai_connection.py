#!/usr/bin/env python3
"""
Script de test pour vérifier la connexion à l'API OpenAI
"""
import os
from dotenv import load_dotenv
import httpx
from openai import OpenAI

# Charger les variables d'environnement
load_dotenv()

def test_connection():
    """Test de connexion à l'API OpenAI"""

    # Vérifier la clé API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Erreur : Clé API OpenAI non trouvée dans .env")
        return False

    print(f"✅ Clé API trouvée : {api_key[:10]}...")

    # Créer le client sans proxy
    print("\n🔧 Configuration du client OpenAI sans proxy...")
    http_client = httpx.Client(
        trust_env=False,  # Ignore les variables d'environnement proxy
        verify=False,      # Désactive la vérification SSL si nécessaire
        timeout=60.0
    )

    client = OpenAI(
        api_key=api_key,
        http_client=http_client
    )

    # Test simple : lister les modèles
    print("🧪 Test de connexion à l'API OpenAI...")
    try:
        models = client.models.list()
        print(f"✅ Connexion réussie ! {len(list(models.data))} modèles disponibles")

        # Vérifier que Whisper est disponible
        model_ids = [m.id for m in models.data]
        if "whisper-1" in model_ids:
            print("✅ Modèle Whisper-1 disponible !")
        else:
            print("⚠️  Modèle Whisper-1 non trouvé dans la liste")

        return True

    except Exception as e:
        print(f"❌ Erreur de connexion : {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TEST DE CONNEXION À L'API OPENAI")
    print("=" * 70)

    if test_connection():
        print("\n" + "=" * 70)
        print("🎉 TOUT FONCTIONNE ! Vous pouvez maintenant utiliser le script principal.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ ÉCHEC DU TEST. Vérifiez votre clé API et votre connexion.")
        print("=" * 70)
