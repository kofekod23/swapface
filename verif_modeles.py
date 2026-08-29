#!/usr/bin/env python3
"""
Vérifie que les modèles ReActor sont présents et non altérés.

Usage :
    python3 verif_modeles.py /chemin/vers/ComfyUI

Les empreintes SHA256 proviennent de la section "Models Hashsum"
du README officiel Gourieff/ComfyUI-ReActor.
"""

import hashlib
import sys
from pathlib import Path

# Empreintes publiées par l'auteur du dépôt.
EMPREINTES = {
    "models/insightface/inswapper_128.onnx":
        "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    "models/insightface/models/buffalo_l/1k3d68.onnx":
        "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "models/insightface/models/buffalo_l/2d106det.onnx":
        "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "models/insightface/models/buffalo_l/det_10g.onnx":
        "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "models/insightface/models/buffalo_l/genderage.onnx":
        "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    "models/insightface/models/buffalo_l/w600k_r50.onnx":
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}

# Fichiers attendus sans empreinte officielle publiée : présence seule.
PRESENCE_SEULE = [
    "models/facerestore_models/codeformer-v0.1.0.pth",
]


def sha256(chemin: Path) -> str:
    """Calcule l'empreinte SHA256 par blocs, sans charger le fichier en mémoire."""
    condensat = hashlib.sha256()
    with chemin.open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
            condensat.update(bloc)
    return condensat.hexdigest()


def taille_lisible(octets: int) -> str:
    for unite in ("o", "Ko", "Mo", "Go"):
        if octets < 1024:
            return f"{octets:.0f} {unite}"
        octets /= 1024
    return f"{octets:.1f} To"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    racine = Path(sys.argv[1]).expanduser().resolve()
    if not racine.is_dir():
        print(f"Dossier introuvable : {racine}")
        return 2

    print(f"Racine ComfyUI : {racine}\n")
    problemes = 0

    for relatif, attendu in EMPREINTES.items():
        chemin = racine / relatif
        if not chemin.is_file():
            print(f"[ABSENT]   {relatif}")
            problemes += 1
            continue

        obtenu = sha256(chemin)
        if obtenu == attendu:
            print(f"[OK]       {relatif}  ({taille_lisible(chemin.stat().st_size)})")
        else:
            print(f"[ALTERE]   {relatif}")
            print(f"           attendu : {attendu}")
            print(f"           obtenu  : {obtenu}")
            problemes += 1

    for relatif in PRESENCE_SEULE:
        chemin = racine / relatif
        if chemin.is_file():
            print(f"[PRESENT]  {relatif}  ({taille_lisible(chemin.stat().st_size)})")
        else:
            print(f"[ABSENT]   {relatif}")
            problemes += 1

    print()
    if problemes:
        print(f"{problemes} probleme(s). Ne lance pas ComfyUI tant que ce n'est pas regle.")
        return 1

    print("Tous les modeles sont en place et conformes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
