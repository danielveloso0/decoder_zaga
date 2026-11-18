# VAE Hurdle–Gamma + cópula: modelo pré-treinado portátil

Esta pasta funciona fora do repositório de pesquisa. Contém **encoder, decoder e pesos treinados da seed0**, cargas da cópula gaussiana de posto 32, ordem dos 32.285 genes e estado de normalização. Não é necessário treinar novamente ou ter os arquivos E8.5/E9.5 para gerar as previsões de referência.

## Instalação em outro projeto

Use Python 3.11 ou 3.12, preferencialmente em ambiente virtual. A reprodução registrada usa Python 3.12/CPU.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest test_portable -v
```

`requirements-tested.txt` registra as versões exatas usadas no teste local. As faixas em `requirements.txt` facilitam a instalação, mas não garantem resultados bit a bit em outros ambientes/BLAS/CPUs. Para reproduzir números, prefira as versões testadas.

## Gerar E10.5 e validar com score_local

Na raiz desta pasta:

```bash
python generate.py --preset V0 --output outputs/prediction_V0.h5ad
python generate.py --preset V3 --output outputs/prediction_V3.h5ad
python generate.py --preset ot_local --output outputs/prediction_ot_local.h5ad
python generate.py --preset ot_ridge --output outputs/prediction_ot_ridge.h5ad
```

Cada preset contém **2.000 latentes E10.5 históricos**, anteriores a esta extração. A matriz de expressão é inteiramente regenerada pelos pesos e pela cópula, não copiada de uma submissão. Isso permite auditar a geração independentemente da pesquisa. O pacote não reajusta GMM/OT e não é um modelo condicionado por tempo; não basta trocar o nome do estágio para prever E12.5.

Os dados reais de referência/alvo **não são incluídos**. Para avaliar, passe seus caminhos locais:

```bash
python score.py --prediction outputs/prediction_ot_ridge.h5ad --e105 /caminho/E10.5_RNA.h5ad --e95 /caminho/E9.5_RNA.h5ad --output outputs/score_ot_ridge.json
```

Em Windows, use caminhos entre aspas quando houver espaços. `score.py` chama o `score_local` preservado e o core de métricas incluído, com atribuição em `evaluation_vendor/NOTICE.md`. Não instala nem atualiza `veckit` global. O scorer usa 1.249 E10.5 alvo, 1.706 E9.5 referência e as 2.000 células previstas (variograma limita 1.500 internamente).

Referências históricas esperadas: V0 **51,5864**, V3 **52,9432**, OT local **55,8939**, OT suavizado/ridge **56,0532**. Esses valores são do protocolo E10.5 de desenvolvimento, não de E12.5. Veja `validation/REPORT.md` para o teste efetivamente realizado nesta distribuição; `validation/expected.json` é a referência anterior, não um novo teste.

Os scripts recusam sobrescrever arquivos. Escolha outro caminho para repetir. Nenhuma execução faz upload.

## Usar o encoder e o decoder no projeto do seu amigo

```python
from hurdle_gamma_copula import PretrainedGenerator

generator = PretrainedGenerator()

# x: células x 32.285, valores log1p(CP10k), genes na ordem exata.
mu, log_variance = generator.encode(x, genes=gene_names)

# Depois de construir seus próprios latentes de 32 dimensões:
expression_csr = generator.decode(z_custom, copula_seed=40000)
generator.write_h5ad(z_custom, "outputs/custom.h5ad")
```

Também é possível codificar um H5AD compatível e decodificar um NPZ pela CLI:

```bash
python encode.py --input meu_dado_logCP10k.h5ad --output outputs/latentes.npz
python generate.py --latent-file outputs/
latentes.npz --latent-key z --output outputs/reconstrucao.h5ad
```

Codificar/decodificar os mesmos dados é reconstrução, não extrapolação. Para extrapolar seus dados, estime uma dinâmica nesse espaço latente e passe o resultado ao decoder. Latentes arbitrários de outro VAE não compartilham necessariamente o mesmo sistema de coordenadas.

Para gerar outro número de células pelo baseline sem deriva:

```bash
python generate.py --bootstrap --cells 5000 --output outputs/baseline_5000.h5ad
```

Esse comando reamostra o banco histórico E9.5 de 2.048 embeddings; não transforma o modelo em uma previsão temporal nova. `decode()` aceita qualquer número positivo de latentes; a escrita com contrato de submissão exige 1.000–5.118 células. Para uso externo sem esse contrato, passe `submission=False` a `write_h5ad()`.

## Conteúdo e limites

- `hurdle_gamma_copula/assets/weights.pt`: somente tensores do VAE; leitura com `weights_only=True`, sem precisar das classes originais ou de objetos Python salvos.
- `decoder_state.npz`: cargas da cópula, embeddings históricos E9.5 e biblioteca; sem índices de células.
- `forecasts.npz`: quatro conjuntos de latentes históricos E10.5 para reprodução.
- `genes.txt`: nomes na ordem obrigatória.
- `config.json`: arquitetura e SHA256 dos assets; verificados ao carregar.
- `PROVENANCE.json`: origem dos arquivos e confirmação da igualdade dos pesos.
- `validation/`: referências e resultados da validação independente de localização.
- `outputs/`: previsões locais, ignoradas pelo Git; não precisa compartilhar.

O checkpoint foi treinado em 2.048 células E8.5 + 2.048 E9.5; validação VAE 128 por estágio; cópula em 512 E9.5 de treino. VAE latente 32/hidden 64, encoder Linear–LayerNorm–SiLU, decoder Hurdle–Gamma. A cópula é ajustada depois, não por NLL conjunta. Os cortes e a normalização final alteram as marginais ideais. Pesos e estados permitem inferência; não há estado de otimizador para retomar exatamente o treino.

Não foram incluídos os arquivos brutos, alvos E10.5, IDs de células, chaves, credenciais ou dependências do ambiente original. **Há estados derivados de E9.5 e previsões latentes**, descritos acima. Aplicação a outra espécie, painel gênico, processamento ou domínio não foi validada. O pacote rejeita ordem incorreta dos genes; não normaliza automaticamente contagens brutas nem faz alinhamento automático.

## Preparar para GitHub

Crie um repositório a partir **somente desta pasta**, preservando `hurdle_gamma_copula/assets/` e os avisos de terceiros. Não é necessário copiar `src/` nem `.gen_deps`. Cada asset está abaixo do limite usual de 100 MB por arquivo. Os resultados H5AD ficam ignorados; os pesos não ficam ignorados. A licença dos componentes próprios deve ser definida pelo responsável pelo projeto; os arquivos de terceiros mantêm sua atribuição e licença própria. Esta entrega apenas prepara os arquivos, sem criar repositório ou publicar dados.

O arquivo `validation/SHA256SUMS.json` permite conferir a cópia distribuída. Preserve também `.gitattributes`: ele impede conversões automáticas de fim de linha que alterariam os hashes, em particular de `genes.txt`. Alterar os pesos, cargas, genes ou latentes invalida os hashes e requer uma nova versão do bundle.
