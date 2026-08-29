#!/usr/bin/env python3
"""
Fait passer ReActor au format CoreML MLProgram sur Apple Silicon.

ReActor demande le provider CoreML sans aucune option (reactor_swapper.py, ligne
38). onnxruntime retombe alors sur le format historique NeuralNetwork, qui
couvre beaucoup moins d'operateurs. Mesure sur M5 :

    hyperswap_1a_256   NeuralNetwork    0 noeud sur CoreML sur 572   156,5 ms
    hyperswap_1a_256   MLProgram        2 partitions sur 44           51,6 ms

Soit un facteur 3. Le gain vient du modele de swap 256 px, qui etait
integralement sur CPU.

Contrepartie mesuree : inswapper_128 passe de 94 a 112 ms sous MLProgram. Si tu
restes sur inswapper_128, ce patch te coute 18 ms par image. Si tu utilises
hyperswap, il t'en fait gagner 105.

Le patch modifie un depot tiers. Un git pull dans ComfyUI-ReActor l'effacera :
relance ce script apres chaque mise a jour.

Usage :
    python3 optimiser_coreml.py              applique
    python3 optimiser_coreml.py --restaurer  remet la ligne d'origine
    python3 optimiser_coreml.py --etat       dit ou on en est
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
CIBLE = (RACINE / "ComfyUI" / "custom_nodes" / "ComfyUI-ReActor"
         / "scripts" / "reactor_swapper.py")

ORIGINE = '        providers = ["CoreMLExecutionProvider"]\n'

REMPLACEMENT = (
    '        # Patch local : MLProgram couvre bien plus d\'operateurs que le\n'
    '        # format NeuralNetwork par defaut. Voir optimiser_coreml.py.\n'
    '        providers = [("CoreMLExecutionProvider", {\n'
    '            "ModelFormat": "MLProgram",\n'
    '            "MLComputeUnits": "CPUAndGPU",\n'
    '        })]\n'
)

MARQUEUR = '"ModelFormat": "MLProgram"'


def lire():
    if not CIBLE.is_file():
        print(f"Fichier introuvable : {CIBLE}", file=sys.stderr)
        sys.exit(2)
    return CIBLE.read_text()


def etat(contenu):
    if MARQUEUR in contenu:
        return "applique"
    if ORIGINE in contenu:
        return "origine"
    return "inconnu"


def main():
    arguments = sys.argv[1:]
    contenu = lire()
    situation = etat(contenu)

    if "--etat" in arguments:
        print(f"{CIBLE}\netat : {situation}")
        return 0

    if "--restaurer" in arguments:
        if situation != "applique":
            print(f"Rien a restaurer, etat : {situation}")
            return 0
        CIBLE.write_text(contenu.replace(REMPLACEMENT, ORIGINE))
        print("Ligne d'origine remise.")
        return 0

    if situation == "applique":
        print("Deja applique, rien a faire.")
        return 0
    if situation == "inconnu":
        print(
            "La ligne attendue n'est pas la et le patch n'est pas present non plus.\n"
            "Le depot a probablement change. Verifie a la main :\n"
            f"  {CIBLE}, bloc PROVIDERS",
            file=sys.stderr,
        )
        return 1

    CIBLE.write_text(contenu.replace(ORIGINE, REMPLACEMENT))
    print(f"Patch applique dans {CIBLE.name}.")
    print("Relance ComfyUI pour qu'il soit pris en compte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
