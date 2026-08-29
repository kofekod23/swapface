#!/usr/bin/env python3
"""
Pilote un serveur ComfyUI depuis ton Mac, en local ou a travers un tunnel.

Le meme script sert les deux cas : c'est la meme API HTTP derriere, seule
l'adresse change.

    # calcul sur ta machine
    python3 piloter.py --video source.mp4 --visage mon_visage.jpg

    # calcul sur Colab, interface et fichiers chez toi
    python3 piloter.py --serveur https://xxx.trycloudflare.com \
                       --video source.mp4 --visage mon_visage.jpg

Le serveur doit tourner avant. En local : python3 lancer.py

Verification de la logique, sans reseau :
    python3 piloter.py --autotest
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent
WORKFLOW = RACINE / "workflow.json"

# Un identifiant stable suffit : ComfyUI s'en sert pour router les evenements.
CLIENT = "piloter-swapface"


def charger_workflow():
    return json.loads(WORKFLOW.read_text())


def regler_workflow(workflow, video, visage, options):
    """Injecte les reglages dans le graphe. Fonction pure, testee par --autotest."""
    workflow["1"]["inputs"]["video"] = video
    workflow["1"]["inputs"]["frame_load_cap"] = options["images"]
    workflow["1"]["inputs"]["skip_first_frames"] = options["depart"]
    workflow["1"]["inputs"]["custom_width"] = options["largeur"]
    workflow["1"]["inputs"]["custom_height"] = options["hauteur"]
    workflow["2"]["inputs"]["image"] = visage
    workflow["3"]["inputs"]["swap_model"] = options["modele"]
    workflow["3"]["inputs"]["face_restore_model"] = options["restauration"]
    workflow["3"]["inputs"]["codeformer_weight"] = options["poids"]
    workflow["4"]["inputs"]["frame_rate"] = options["fps"]
    # Une source sans piste audio fait echouer la sortie audio de VHS_LoadVideo.
    # On coupe alors la liaison plutot que de laisser le rendu planter.
    if options.get("sans_audio"):
        workflow["4"]["inputs"].pop("audio", None)
    return workflow


def images_par_seconde(chemin):
    """Lit la cadence de la video avec le ffmpeg embarque par imageio-ffmpeg.
    Retourne None si la lecture echoue, l'appelant decide alors."""
    try:
        import imageio_ffmpeg
        binaire = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    try:
        sortie = subprocess.run(
            [binaire, "-i", str(chemin)],
            capture_output=True, text=True, timeout=60,
        ).stderr
    except Exception:
        return None
    trouve = re.search(r"([0-9]+(?:\.[0-9]+)?) fps", sortie)
    return float(trouve.group(1)) if trouve else None


def a_une_piste_audio(chemin):
    """Dit si la video porte une piste audio. En cas de doute, repond oui :
    le graphe complet est le cas normal, on ne coupe l'audio que sur preuve."""
    try:
        import imageio_ffmpeg
        binaire = imageio_ffmpeg.get_ffmpeg_exe()
        sortie = subprocess.run(
            [binaire, "-i", str(chemin)],
            capture_output=True, text=True, timeout=60,
        ).stderr
    except Exception:
        return True
    return bool(re.search(r"Stream #\d+:\d+.*: Audio:", sortie))


def televerser(session, serveur, chemin):
    """Envoie un fichier dans le dossier input du serveur, retourne son nom."""
    chemin = Path(chemin).expanduser().resolve()
    if not chemin.is_file():
        print(f"Fichier introuvable : {chemin}", file=sys.stderr)
        sys.exit(2)
    with chemin.open("rb") as fichier:
        reponse = session.post(
            f"{serveur}/upload/image",
            files={"image": (chemin.name, fichier, "application/octet-stream")},
            data={"type": "input", "overwrite": "true"},
            timeout=600,
        )
    reponse.raise_for_status()
    nom = reponse.json()["name"]
    print(f"  televerse : {nom}")
    return nom


def suivre(session, serveur, identifiant):
    """Attend la fin du rendu en interrogeant l'historique. Retourne la sortie."""
    debut = time.time()
    dernier_reste = None
    while True:
        historique = session.get(f"{serveur}/history/{identifiant}", timeout=60).json()
        if identifiant in historique:
            return historique[identifiant]

        try:
            reste = session.get(f"{serveur}/prompt", timeout=30).json().get("exec_info", {}).get("queue_remaining")
        except Exception:
            reste = None
        if reste != dernier_reste:
            dernier_reste = reste
            print(f"  en cours, file : {reste}, {time.time() - debut:.0f} s ecoulees")
        time.sleep(2)


def recuperer(session, serveur, sortie, destination):
    """Telecharge le fichier produit par VHS_VideoCombine."""
    fichiers = []
    for donnees in sortie.get("outputs", {}).values():
        fichiers.extend(donnees.get("gifs", []))
    if not fichiers:
        print("Aucun fichier produit. Sortie brute :", file=sys.stderr)
        print(json.dumps(sortie.get("outputs", {}), indent=2)[:2000], file=sys.stderr)
        return None

    fichier = fichiers[-1]
    reponse = session.get(
        f"{serveur}/view",
        params={
            "filename": fichier["filename"],
            "subfolder": fichier.get("subfolder", ""),
            "type": fichier.get("type", "output"),
        },
        timeout=600,
    )
    reponse.raise_for_status()
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(reponse.content)
    return destination


def autotest():
    """Verifie que les reglages atterrissent aux bons endroits du graphe."""
    workflow = regler_workflow(
        charger_workflow(),
        "a.mp4",
        "b.jpg",
        {"images": 12, "depart": 90, "largeur": 640, "hauteur": 360,
         "modele": "inswapper_128.onnx",
         "restauration": "codeformer-v0.1.0.pth", "poids": 0.4, "fps": 30.0},
    )
    assert workflow["1"]["inputs"]["video"] == "a.mp4"
    assert workflow["1"]["inputs"]["frame_load_cap"] == 12
    assert workflow["1"]["inputs"]["skip_first_frames"] == 90
    assert workflow["1"]["inputs"]["custom_width"] == 640
    assert workflow["2"]["inputs"]["image"] == "b.jpg"
    assert workflow["3"]["inputs"]["swap_model"] == "inswapper_128.onnx"
    assert workflow["3"]["inputs"]["codeformer_weight"] == 0.4
    assert workflow["4"]["inputs"]["frame_rate"] == 30.0
    # Les liaisons ne doivent pas avoir bouge.
    assert workflow["3"]["inputs"]["input_image"] == ["1", 0]
    assert workflow["3"]["inputs"]["source_image"] == ["2", 0]
    assert workflow["4"]["inputs"]["images"] == ["3", 0]
    assert workflow["4"]["inputs"]["audio"] == ["1", 2]

    muet = regler_workflow(
        charger_workflow(), "a.mp4", "b.jpg",
        {"images": 5, "depart": 0, "largeur": 640, "hauteur": 360,
         "modele": "inswapper_128.onnx",
         "restauration": "none", "poids": 0.7, "fps": 25.0, "sans_audio": True},
    )
    assert "audio" not in muet["4"]["inputs"]
    assert muet["4"]["inputs"]["images"] == ["3", 0]
    print("autotest OK")
    return 0


def main():
    analyseur = argparse.ArgumentParser(
        description="Pilote ComfyUI pour un remplacement de visage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analyseur.add_argument("--autotest", action="store_true",
                           help="verifie la logique sans reseau puis sort")
    analyseur.add_argument("--serveur", default="http://127.0.0.1:8188",
                           help="adresse du serveur ComfyUI")
    analyseur.add_argument("--video", help="video source")
    analyseur.add_argument("--visage", help="image du visage a poser")
    analyseur.add_argument("--sortie", default="rendu.mp4", help="fichier a ecrire")
    analyseur.add_argument("--modele", default="hyperswap_1a_256.onnx",
                           help="hyperswap_1a_256.onnx ou inswapper_128.onnx")
    analyseur.add_argument("--restauration", default="none",
                           help="none ou codeformer-v0.1.0.pth")
    analyseur.add_argument("--poids", type=float, default=0.7,
                           help="codeformer_weight, entre 0 et 1")
    analyseur.add_argument("--images", type=int, default=30,
                           help="frame_load_cap, 0 pour tout charger")
    analyseur.add_argument("--depart", type=int, default=0,
                           help="skip_first_frames, image de depart")
    analyseur.add_argument("--largeur", type=int, default=1280)
    analyseur.add_argument("--hauteur", type=int, default=720)
    analyseur.add_argument("--fps", type=float, default=None,
                           help="cadence de sortie, lue dans la source par defaut")
    analyseur.add_argument("--sans-audio", action="store_true",
                           help="ne pas reprendre l'audio de la source, "
                                "obligatoire si la source n'a pas de piste audio")
    arguments = analyseur.parse_args()

    if arguments.autotest:
        return autotest()

    if not (arguments.video and arguments.visage):
        analyseur.error("--video et --visage sont obligatoires")

    import requests

    serveur = arguments.serveur.rstrip("/")
    session = requests.Session()

    print(f"Serveur : {serveur}")
    try:
        session.get(f"{serveur}/system_stats", timeout=30).raise_for_status()
    except Exception as erreur:
        print(f"Serveur injoignable : {erreur}", file=sys.stderr)
        print("En local, demarre-le avec : python3 lancer.py", file=sys.stderr)
        return 2

    audio_present = a_une_piste_audio(arguments.video)
    if not audio_present:
        print("  la source n'a pas de piste audio, le rendu sera muet")
    fps = arguments.fps or images_par_seconde(arguments.video)
    if fps is None:
        fps = 25.0
        print("  cadence illisible, repli sur 25 images par seconde")
    else:
        print(f"  cadence source : {fps} images par seconde")

    print("Televersement")
    nom_video = televerser(session, serveur, arguments.video)
    nom_visage = televerser(session, serveur, arguments.visage)

    workflow = regler_workflow(
        charger_workflow(), nom_video, nom_visage,
        {"images": arguments.images, "depart": arguments.depart,
         "largeur": arguments.largeur,
         "hauteur": arguments.hauteur, "modele": arguments.modele,
         "restauration": arguments.restauration, "poids": arguments.poids,
         "fps": fps, "sans_audio": arguments.sans_audio or not audio_present},
    )

    print(f"Rendu : {arguments.modele}, restauration {arguments.restauration}, "
          f"{arguments.images or 'toutes les'} images, {arguments.largeur}x{arguments.hauteur}")
    reponse = session.post(f"{serveur}/prompt",
                           json={"prompt": workflow, "client_id": CLIENT}, timeout=120)
    if reponse.status_code != 200:
        print("Le serveur a refuse le graphe :", file=sys.stderr)
        print(json.dumps(reponse.json(), indent=2, ensure_ascii=False)[:3000], file=sys.stderr)
        return 1

    identifiant = reponse.json()["prompt_id"]
    debut = time.time()
    resultat = suivre(session, serveur, identifiant)
    duree = time.time() - debut

    destination = recuperer(session, serveur, resultat, arguments.sortie)
    if destination is None:
        return 1

    images = arguments.images or 0
    par_image = f", {1000 * duree / images:.0f} ms par image" if images else ""
    print(f"\nEcrit : {destination}")
    print(f"Duree : {duree:.1f} s{par_image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
