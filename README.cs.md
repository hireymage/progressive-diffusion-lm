# Diffuzní jazykový model s progresivní přesností

[English](README.md) | [Čeština](README.cs.md)

> **Soukromý source-only staging snapshot.** Výsledky a dokumentace pro veřejné vydání se stále ověřují.

Výzkumný prototyp pro Apple MLX testuje, zda mohou časné denoising kroky s vysokým šumem používat nižší přesnost vah než pozdní jemné kroky bez významného zhoršení kvality.

Snapshot obsahuje zdrojový kód, konfigurace, testy, campaign tooling a licenci. Záměrně neobsahuje datasety, tokenizer artefakty, cache, checkpointy, runtime logy ani výsledkové bundly.

Dřívější výsledky procházejí novým auditem po nalezení reprodukčních chyb v původním kódu. Dokud nebudou zveřejněné opravené manifests, repository **netvrdí, že progressive precision, Q1 nebo jiný schedule překonává FP32**.

## Implementace

- Masked/absorbing diffusion a bidirectional Transformer.
- FP32 master weights; Q1–Q4 jsou simulované v FP32 pomocí STE.
- Index schedule `0` odpovídá nejvyššímu šumu/coarse kroku, poslední index nejnižšímu šumu/fine kroku.
- `bits=0` je optional ternary, `1` Q1, `2` Q2, `3` Q3, `4` Q4 a `16` identity/FP32.
- FP32 baseline používá 32 storage bitů. Průměrná bitová šířka schedule je časová výpočetní charakteristika, nikoli velikost uloženého modelu.
- Nejsou implementované packed integer weights ani low-bit kernels; aktuálně tedy nevzniká reálná úspora velikosti ani zrychlení.

## Rychlý start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest -q
./run_smoke_test.sh
```

Checked-in experimentální konfigurace používají `dropout=0.0` a `weight_decay=0.0`, což odpovídá skutečnému historickému protokolu. Kód nyní podporuje attention dropout a AdamW weight decay, ale jejich zapnutí představuje jiný experimentální protokol.

Cache datasetu je identifikována názvem/configem/revizí datasetu, SHA-256 tokenizeru, splitem, seedem a limity. Workflow nyní používá pinned revizi `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` a ukládá corpus provenance i k tokenizeru. Historická data nemají prokazatelnou immutable upstream revizi, pouze zachované checksumy.

Porovnávací evaluace používá pro oba modely shodné batchy a corruption masky. Checkpoint restart obnovuje model, optimizer a vlastní step metadata, ale nikoli všechny iterátory, RNG a historii metrik; jde proto o **warm restart**, ne bitově přesné pokračování.

Poslední lokální kontrola opraveného source-only stromu měla `132 passed`.

## Hlavní omezení

- malá výzkumná architektura a omezený dataset;
- simulated quantization bez skutečných low-bit kernelů;
- historické nonconstant schedule výsledky potřebují opravené směrové popisky;
- staré runs byly ve skutečnosti unregularized Adam, přestože config uváděl dropout/weight decay;
- žádné obecné tvrzení o výhodě Q1 nebo progressive precision zatím není oprávněné.

## Licence

Vlastní zdrojový kód je pod licencí [Apache License 2.0](LICENSE). Externí datasety, závislosti, tokenizery a cizí modely si zachovávají své licence.
