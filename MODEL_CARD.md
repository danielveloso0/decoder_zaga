# Modelo e contrato de entrada

**Finalidade:** representação e geração probabilística de expressão unicelular na escala log1p(CP10k), com pesos históricos da linha generativa Virtual Embryo. Modelo seed0, checkpoint selecionado na época27.

**Arquitetura:** encoder 32.285→64, LayerNorm/SiLU, cabeças média/log-variância64→32; decoder32→64/ReLU, cabeças probabilidade e média positiva64→32.285; forma Gamma aprendida por gene. Cópula residual gaussiana de posto32 ajustada depois do VAE. Preservamos o código original de modelo e sampler.

**Dados:** VAE treinado em2.048 E8.5+2.048 E9.5; validação128 por estágio. Cópula ajustada em512 E9.5 de treino. Sem rótulos celulares. Estados derivados históricos são incluídos; dados brutos e identificadores não.

**Entrada obrigatória:** mesmos32.285 genes e ordem de assets/genes.txt; valores finitos não negativos já em log1p(CP10k). O pacote não reconhece automaticamente escala, espécie, dados brutos ou mudança de domínio. encode() retorna médias posteriores e log-variâncias; um encoder não prevê tempo.

**Saída:** matriz CSR float32, zeros e positivos, biblioteca linear10.000 após inversão log1p/reescala. A Gamma é definida sobre positivos de log-expressão antes dos cortes e da normalização final. Cópula modifica dependência durante o sorteio; não é aplicada depois a uma expressão independente.

**Referências:** redes Bernoulli–Gamma: Cannon2008, DOI10.1175/2008JHM960.1; VAE: Kingma/Welling, arXiv1312.6114; resíduos aleatorizados: Dunn/Smyth1996, DOI10.1080/10618600.1996.10474708; cópula gaussiana fatorial: Murray etal.2013, DOI10.1080/01621459.2012.762328. A arquitetura exata, escala Gamma, hiperparâmetros, SVD plug-in e combinação dessas ferramentas são escolhas do projeto, não reprodução integral desses artigos.

**Validação:** ver validation/REPORT.md. Os presets congelados validam portabilidade/reprodução da geração, não aprendizado temporal independente. E10.5 já foi usado para desenvolvimento. Não há validação de E12.5 ou de dados de terceiros.

**Limites:** não há estado de otimizador ou pipeline completa de treino; não inclui ajuste de novas dinâmicas. Pesos são congelados pela API; model.py mantém a formulação do objetivo original. Amostrar N(0,I) no lugar das distribuições históricas não foi validado como previsão. CPU foi testada; alterar blocos, seeds ou bibliotecas pode mudar a reprodução.

**Publicação:** nenhum repositório foi criado/enviado. Código/pesos próprios sem nova licença atribuída nesta extração. Avisos/licenças de terceiros permanecem em evaluation_vendor.
