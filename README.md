# ResNet (PyTorch) — Paper-Faithful Implementation

Implementation of *Deep Residual Learning for Image Recognition*
(He et al., 2015) with a CIFAR-10 training script.

## Models

| family | models | blocks | shortcut default |
|--------|--------|--------|------------------|
| ImageNet (Table 1) | `resnet18/34/50/101/152` | BasicBlock / Bottleneck | B (projection when dims change) |
| CIFAR-10 (Sec. 4.2) | `resnet20/32/44/56/110/1202` | BasicBlock, 3 stages of 16/32/64 | A (zero-pad, parameter-free) |
| Plain baseline (Fig. 6) | `plain20/32/44/56/110` | aynı topoloji, **skip connection yok** | — |

Plain modeller residual bağlantı olmadan aynı ağdır (option A parametre
içermediği için parametre sayıları ResNet'le birebir aynı). Amaç makalenin
ana motivasyonunu (degradation problem) yeniden üretmek: derinlik arttıkça
plain ağların hatası **artar**, ResNet'lerin hatası **düşer**.

- Shortcut options **A** (zero-padding identity), **B** (projection when dims
  change), **C** (projection everywhere) — selectable via `--shortcut`
- ImageNet models also accept `--stem cifar` (3×3 s1 stem, no maxpool) to run
  on 32×32 input
- Kaiming/He normal init for conv & linear weights, BN γ=1, β=0

## Train on CIFAR-10

```bash
pip install -r requirements.txt
python train.py resnet20                      # paper recipe
python train.py plain20                       # residual'sız baseline
python train.py resnet110 --warmup-epochs 1   # paper warms up 110/1202
python train.py resnet34 --stem cifar         # ImageNet net on CIFAR
python train.py resnet38                      # herhangi bir 6n+2 derinlik
```

Model adı pozisyonel: `resnet<depth>` veya `plain<depth>` yaz, geçerli her
6n+2 derinlik (20, 26, 32, 38, 44, ...) isimden otomatik üretilir. Geçersiz
derinlikte (örn. `resnet36`) en yakın geçerli derinlikler önerilir.

Defaults follow the paper's recipe: SGD momentum 0.9, weight decay 1e-4,
batch 128, LR 0.1 ÷10 at epochs 82/123 (≈32k/48k iters), 164 epochs (≈64k
iters), 4-pixel pad + random crop + horizontal flip.

Dataset için bir şey yapmana gerek yok: ilk çalıştırmada `torchvision`
CIFAR-10'u (~170MB) `--data-dir` altına otomatik indirir.

## RunPod'da eğitim

1. [runpod.io](https://runpod.io) → Pods → Deploy: bir GPU seç (CIFAR ResNet'leri
   küçüktür; RTX 3090/4090 fazlasıyla yeter) ve **RunPod PyTorch 2.x**
   template'ini kullan.
2. Pod açılınca web terminal veya SSH ile bağlan:

```bash
cd /workspace                       # kalıcı disk; pod durunca silinmez
git clone <your-repo-url> resnet && cd resnet
bash scripts/runpod_setup.sh        # bağımlılık + sanity check + dataset indirme
tmux new -s train                   # bağlantı kopsa da eğitim sürsün
bash scripts/train_all.sh           # plain20→56 + resnet20→110'u sırayla eğitir
```

3. Sonuçlar `logs/` altına yazılır: her model için epoch-bazlı `<model>.csv`,
   özet `<model>.json` ve tam konsol çıktısı `<model>.train.log`.
   Makale (Table 6) ile karşılaştırma:

```bash
python compare_results.py
```

tmux'tan `Ctrl+B D` ile çıkıp `tmux attach -t train` ile geri dönebilirsin.
İşin bitince podu **Stop** et (GPU ücreti durur); `/workspace` kalıcıdır.

Hızlı varyasyonlar:

```bash
MODELS="resnet20 resnet56" bash scripts/train_all.sh   # sadece bu ikisi
EPOCHS=82 bash scripts/train_all.sh                    # yarım schedule, hızlı deneme
python train.py resnet110 --seed 1 --run-name resnet110_s1  # farklı seed
```

## Beklenen sonuçlar (paper Table 6, test error %)

| model | paper err% |
|-------|-----------|
| resnet20 | 8.75 |
| resnet32 | 7.51 |
| resnet44 | 7.17 |
| resnet56 | 6.97 |
| resnet110 | 6.43 (5 koşunun en iyisi; ort. 6.61±0.16) |
| resnet1202 | 7.93 (overfit, paper de raporluyor) |

Plain ağlar için makale tablo vermiyor (yalnızca Fig. 6 eğrileri); beklenen
davranış hatanın derinlikle artması — kabaca plain20 ≈ %9-10'dan plain56'da
%12-13'e doğru — yani ResNet'lerin tam tersi yönde bir eğri.

## Useful flags

| flag | default | notes |
|------|---------|-------|
| `model` (pozisyonel) | `resnet20` | `resnet<6n+2>` / `plain<6n+2>` / ImageNet ailesi |
| `--shortcut` | model default | `A` / `B` / `C` |
| `--epochs` | `164` | paper: 64k iter |
| `--batch-size` | `128` | |
| `--lr` | `0.1` | SGD, momentum 0.9, wd 1e-4 |
| `--milestones` | `82 123` | LR /= 10 |
| `--warmup-epochs` | `0` | resnet110/1202 için `1` önerilir |
| `--stem` | `cifar` | sadece ImageNet ailesi için |

## Tahmin / test

Eğitilmiş bir checkpoint ile test setinden rastgele resimleri sınıflandırıp
confidence ile basar, `predictions.png` olarak grid kaydeder (yeşil başlık =
doğru, kırmızı = yanlış):

```bash
python predict.py checkpoints/resnet20_best.pt
python predict.py checkpoints/resnet20_best.pt --num-images 16 --seed 42
```

Sanity check — tüm modellerin parametre sayısını ve forward shape'ini basar
(CIFAR modelleri makaledekiyle birebir: 0.27M / 0.46M / 0.66M / 0.85M / 1.7M / 19.4M):

```bash
python model.py
```
