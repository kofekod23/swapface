# swapface

Remplacement de visage sur video, avec ComfyUI et ReActor, sur macOS Apple
Silicon ou sur GPU distant.

Ce depot ne contient que les scripts. ComfyUI, ReActor, VideoHelperSuite et les
modeles sont installes a cote et restent hors de git.

## Ce que ca fait

```
VHS_LoadVideo (source.mp4)
  IMAGE  -> ReActorFaceSwap.input_image
  AUDIO  -> VHS_VideoCombine.audio

LoadImage (mon_visage.jpg)
  IMAGE  -> ReActorFaceSwap.source_image

ReActorFaceSwap
  IMAGE  -> VHS_VideoCombine.images
```

Deux facons de s'en servir, avec le meme graphe :

- **l'interface web de ComfyUI**, pour tatonner sur les reglages ;
- **`piloter.py`**, en ligne de commande, pour un rendu reproductible.

Le meme `piloter.py` vise ton Mac ou un serveur distant, seule l'option
`--serveur` change.

## Installation sur macOS Apple Silicon

Il te faut `git` et [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kofekod23/swapface.git
cd swapface

uv python install 3.12
uv venv --python 3.12 venv
VIRTUAL_ENV=$PWD/venv uv pip install pip setuptools importlib_metadata

git clone https://github.com/Comfy-Org/ComfyUI.git
VIRTUAL_ENV=$PWD/venv uv pip install -r ComfyUI/requirements.txt

git clone -b main https://github.com/Gourieff/ComfyUI-ReActor.git \
    ComfyUI/custom_nodes/ComfyUI-ReActor
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
VIRTUAL_ENV=$PWD/venv uv pip install \
    -r ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt

cd ComfyUI/custom_nodes/ComfyUI-ReActor && ../../../venv/bin/python install.py && cd -
venv/bin/python telecharger_modeles.py --hyperswap
venv/bin/python optimiser_coreml.py
venv/bin/python verif_modeles.py ComfyUI
```

`verif_modeles.py` doit afficher sept lignes vertes. Sinon, ne va pas plus loin.

## Usage

```bash
# un terminal
python3 lancer.py

# un autre
python3 piloter.py --video source.mp4 --visage mon_visage.jpg --images 0
```

L'interface web est sur http://127.0.0.1:8188.

### Options de piloter.py

| option | defaut | role |
|---|---|---|
| `--serveur` | `http://127.0.0.1:8188` | adresse de ComfyUI, locale ou tunnel |
| `--video` | | video source |
| `--visage` | | image du visage a poser |
| `--sortie` | `rendu.mp4` | fichier a ecrire |
| `--modele` | `hyperswap_1a_256.onnx` | ou `inswapper_128.onnx` |
| `--restauration` | `none` | ou `codeformer-v0.1.0.pth` |
| `--poids` | `0.7` | `codeformer_weight` |
| `--images` | `30` | nombre d'images, `0` pour toute la video |
| `--depart` | `0` | image de depart |
| `--largeur` `--hauteur` | `1280` `720` | `0` `0` garde la resolution native |
| `--fps` | lue dans la source | cadence de sortie |
| `--sans-audio` | | detecte tout seul, force le rendu muet |

`--largeur 0 --hauteur 0` est presque toujours le bon choix : agrandir une source
360p avant le swap ne cree aucun detail.

## Choisir un plan

Un remplacement rate n'est pas toujours un probleme de reglage. Sur un plan de
profil ou de dos, le modele n'a pas les reperes qu'il lui faut.

```bash
python3 trouver_plan.py source.mp4
```

Il analyse la video avec le detecteur de ReActor et te donne les passages ou le
visage est grand et de face, avec la commande `piloter.py` correspondante.

## Sur GPU distant

`colab_swapface.ipynb` installe tout sur Colab, demarre ComfyUI et ouvre un
tunnel cloudflared. Tu recuperes une URL publique :

```bash
python3 piloter.py --serveur https://xxx.trycloudflare.com \
                   --video source.mp4 --visage mon_visage.jpg --images 0
```

[Ouvrir dans Colab](https://colab.research.google.com/github/kofekod23/swapface/blob/main/colab_swapface.ipynb)

**Ce n'est pas forcement plus rapide.** Voir la section suivante.

## Performances mesurees

Sur MacBook Pro M5, 32 Go, macOS 26.5.2, clip 640x360, un visage par image.

### Temps d'inference des modeles

```
modele                             CoreML          CPU      gain
----------------------------------------------------------------
detection  det_10g                   44,8         55,7     x1,24
identite   w600k_r50                  5,2         29,3     x5,58
swap       inswapper_128            107,2        369,1     x3,44
swap       hyperswap_1a_256          46,7        159,8     x3,42

restauration CodeFormer, sur MPS    355,5
```

Reproduire : `python3 bench.py`

### Placement sur Metal

```
modele                           CoreML      CPU   part GPU
------------------------------------------------------------
det_10g                               4       11       27 %
w600k_r50                             1        1       50 %
inswapper_128                         1        0      100 %
hyperswap_1a_256                      2       42        5 %

CodeFormer sous MPS : aucun repli, 100 % Metal
```

Reproduire : `python3 diag_gpu.py`

ReActor demande le provider CoreML sans options, et onnxruntime retombe alors
sur le format `NeuralNetwork`, qui couvre bien moins d'operateurs. Dans cet etat
`hyperswap_1a_256` tournait entierement sur CPU, 572 noeuds sur 572, a 156,5 ms.
`optimiser_coreml.py` impose `MLProgram` et `CPUAndGPU` : 46,7 ms, un facteur 3.

Ce patch modifie un depot tiers. Relance-le apres chaque mise a jour de
ComfyUI-ReActor. `--etat` dit ou tu en es, `--restaurer` annule.

### Rendu complet, de bout en bout

60 images, 640x360, `hyperswap_1a_256`, mesures a chaud.

| machine | sans restauration | avec CodeFormer 0,7 |
|---|---|---|
| MacBook Pro M5 | 369 ms par image | 703 ms par image |
| Colab A100 40 Go | 1084 ms par image | 1153 ms par image |

Ces chiffres Colab sont **faux**, et la raison merite d'etre connue.

Pendant ces rendus le GPU de la session n'a montre aucune activite. Le paquet
`onnxruntime-gpu` etait pourtant bien installe, et `get_available_providers()`
retournait bien :

```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

Mais cette fonction liste ce que la roue **sait faire**, pas ce qui **s'initialise**.
En creant reellement une session, le verdict tombe :

```
providers effectifs : ['CPUExecutionProvider']
396,3 ms par inference
```

`CUDAExecutionProvider` echoue a l'initialisation et onnxruntime bascule sur le
processeur sans lever d'erreur. Sur Colab, la cause est que `onnxruntime-gpu`
cherche `libcudnn` et `libcublas` dans les chemins systeme, alors qu'elles y
arrivent par les paquets pip de PyTorch, dans `site-packages/nvidia/*/lib`.
Le notebook transmet desormais ces dossiers a ComfyUI par `LD_LIBRARY_PATH`.

C'est exactement le meme piege que CoreML sur Mac, ou le provider etait annonce,
accepte, et ne prenait aucun noeud sur `hyperswap_1a_256`. **La seule preuve est
le temps mesure.** `diag_gpu.py` fait cette verification sur les deux plateformes.

Ce qui reste vrai sans dependre de ce point : le decodage video, le recadrage,
le collage et l'encodage tournent sur le processeur quoi qu'il arrive, et les
deux vCPU d'une session Colab sont plus lents que les dix coeurs d'une M5. Le
GPU distant n'aide que sur la part modele.

## Reglages

- `hyperswap_1a_256.onnx` sort en 256 px natif contre 128 pour `inswapper_128`.
  Depuis le patch CoreML, il est aussi le plus rapide des deux sur Mac.
- `codeformer_weight` autour de 0,7. A 1,0 on gagne en nettete et on perd en
  ressemblance.
- `face_restore_model` a `none` pour les passes de travail. C'est le poste le plus
  lourd sur Mac, tu le rallumes pour le rendu final.

## Les scripts

| fichier | role |
|---|---|
| `lancer.py` | demarre ComfyUI, pose `PYTORCH_ENABLE_MPS_FALLBACK=1`, confirme que le peripherique est `mps` |
| `piloter.py` | televerse, lance le rendu, suit l'avancement, rapatrie le fichier |
| `trouver_plan.py` | repere les passages ou le visage est de face |
| `verif_modeles.py` | compare les SHA256 aux empreintes du README de ReActor |
| `telecharger_modeles.py` | recupere ce que `install.py` ne pose pas |
| `optimiser_coreml.py` | patch `MLProgram`, applique, restaure, ou dit son etat |
| `bench.py` | temps d'inference, CoreML contre CPU, CodeFormer sur MPS |
| `diag_gpu.py` | quels noeuds tournent vraiment sur l'accelerateur, Metal ou CUDA |
| `workflow.json` | le graphe au format API |
| `colab_swapface.ipynb` | installation, GPU et tunnel cote Colab |

Chaque script a une aide : `python3 <script>.py --help`. `lancer.py` et
`piloter.py` ont un `--autotest` qui verifie leur logique sans reseau.

## Pieges rencontres

- `uv venv` ne pose pas `pip`, et `install.py` de ReActor appelle
  `sys.executable -m pip`. Il faut installer `pip` dans le venv.
- `setuptools` 81 et suivants ont supprime `pkg_resources`. `install.py` se rabat
  sur `importlib_metadata`, qu'il faut donc installer. Son test « deja installe »
  echoue alors silencieusement, il reinstalle a chaque passage. Sans consequence.
- `install.py` ne telecharge que `inswapper_128.onnx`. `buffalo_l` et
  `codeformer-v0.1.0.pth` sont a recuperer a part, c'est le role de
  `telecharger_modeles.py`.
- `force_size` n'existe pas sur `VHS_LoadVideo` dans les versions actuelles. Le
  redimensionnement passe par `custom_width` et `custom_height`, et `0` signifie
  « ne pas redimensionner ».
- Une source sans piste audio faisait planter le rendu entier, `VHS_LoadVideo`
  echouant a extraire une piste inexistante. `piloter.py` detecte le cas et coupe
  la liaison.
- Il n'existe aucun drapeau `--mps` ni `--use-mps` dans `comfy/cli_args.py`.
  ComfyUI detecte Metal tout seul.

## Licences des modeles

`inswapper_128.onnx` et `buffalo_l` viennent d'InsightFace et sont sous licence
**non commerciale**. `codeformer-v0.1.0.pth` egalement. Les modeles HyperSwap
viennent de FaceFusion Labs. Verifie les conditions avant tout usage autre que
personnel.

Ce depot sert a remplacer son propre visage sur ses propres videos.
