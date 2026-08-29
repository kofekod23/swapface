#!/usr/bin/env python3
"""
Telecharge les modeles que install.py ne pose pas.

install.py ne recupere que inswapper_128.onnx. buffalo_l est cense arriver au
premier demarrage de ComfyUI, et codeformer-v0.1.0.pth est un telechargement
manuel (README ReActor, lignes 203-204 et 268).

Usage :
    python3 telecharger_modeles.py [--hyperswap]

    --hyperswap  ajoute hyperswap_1a_256.onnx (403 Mo), modele de swap 256 px
                 de FaceFusion Labs, deux fois la resolution native
                 d'inswapper_128.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent
MODELES = RACINE / "ComfyUI" / "models"

HF_REACTOR = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models"
HF_FACEFUSION = "https://huggingface.co/facefusion/models-3.3.0/resolve/main"

# (url, destination relative a ComfyUI/models)
FICHIERS = [
    (f"{HF_REACTOR}/facerestore_models/codeformer-v0.1.0.pth",
     "facerestore_models/codeformer-v0.1.0.pth"),
]

HYPERSWAP = [
    (f"{HF_FACEFUSION}/hyperswap_1a_256.onnx", "hyperswap/hyperswap_1a_256.onnx"),
]

# Archive a decompresser : (url, dossier de destination, fichier temoin)
ARCHIVES = [
    (f"{HF_REACTOR}/buffalo_l.zip",
     "insightface/models/buffalo_l",
     "det_10g.onnx"),
]


def barre(recus, taille_bloc, total):
    if total <= 0:
        return
    fait = min(recus * taille_bloc, total)
    pourcent = 100 * fait / total
    largeur = 30
    plein = int(largeur * fait / total)
    sys.stdout.write(
        f"\r  [{'#' * plein}{'.' * (largeur - plein)}] {pourcent:5.1f} %"
        f"  {fait / 1e6:7.1f} / {total / 1e6:.1f} Mo"
    )
    sys.stdout.flush()


def telecharger(url, destination):
    """Telecharge vers un fichier temporaire puis renomme, pour ne jamais laisser
    un fichier partiel qui passerait pour complet."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partiel = destination.with_suffix(destination.suffix + ".partiel")
    print(f"{destination.name}")
    urllib.request.urlretrieve(url, partiel, reporthook=barre)
    print()
    partiel.replace(destination)


def main():
    a_faire = list(FICHIERS)
    if "--hyperswap" in sys.argv[1:]:
        a_faire += HYPERSWAP

    if not MODELES.is_dir():
        print(f"Dossier introuvable : {MODELES}", file=sys.stderr)
        return 2

    for url, relatif in a_faire:
        destination = MODELES / relatif
        if destination.is_file():
            print(f"{destination.name} : deja present, ignore")
            continue
        telecharger(url, destination)

    for url, dossier_relatif, temoin in ARCHIVES:
        dossier = MODELES / dossier_relatif
        if (dossier / temoin).is_file():
            print(f"{dossier.name} : deja present, ignore")
            continue
        archive = MODELES / Path(url).name
        telecharger(url, archive)
        print(f"  decompression vers {dossier}")
        dossier.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for membre in zf.namelist():
                nom = Path(membre).name
                # L'archive peut contenir ou non un dossier buffalo_l a la racine.
                if not nom or membre.endswith("/"):
                    continue
                (dossier / nom).write_bytes(zf.read(membre))
        archive.unlink()

    print("\nTermine. Lance maintenant : python3 verif_modeles.py ComfyUI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
