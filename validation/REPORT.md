# Validação independente da pasta portátil

Cópia executada fora do repositório: `C:\Users\danie\AppData\Local\Temp\hurdle_copula_isolated_vqrd9ro4\project`. Python com `-I`, sem PYTHONPATH e sem imports de `src`. As bibliotecas de terceiros foram reutilizadas do ambiente existente; não foi criado/instalado um ambiente limpo via pip.

Os pesos portáteis são tensores idênticos aos originais. As matrizes CSR regeneradas têm hash idêntico às submissões originais, e o scorer foi executado novamente nos arquivos novos. Nenhum treino ou upload.

| Preset | Score recalculado | DE | Direção | MMD | Variograma | CSR original |
|---|---:|---:|---:|---:|---:|---|
| V0 | 51.586442 | 46.692779 | 56.234875 | 48.314197 | 56.801347 | Idêntico |
| V3 | 52.943171 | 48.811995 | 57.336720 | 50.062949 | 56.935538 | Idêntico |
| ot_local | 55.893940 | 49.237929 | 60.010633 | 55.206164 | 60.099751 | Idêntico |
| ot_ridge | 56.053171 | 49.237929 | 60.056248 | 55.646516 | 60.178359 | Idêntico |

Alvo: 1.249 E10.5; referência: 1.706 E9.5; previsão: 2.000 células × 32.285 genes. As quatro métricas brutas e os quatro skills coincidiram com os resultados locais anteriores. Isso valida a extração e reprodução, não generalização para dados novos/E12.5.

Testes rápidos: repetibilidade do sampler, dimensões, valores finitos/não negativos, biblioteca10.000, rejeição de genes fora de ordem, latentes inválidos, encode e variância normal unitária da cópula.

O teste usa CPU e versões registradas em requirements-tested.txt. Outro ambiente pode produzir pequenas diferenças numéricas. Ver logs de imports isolados e JSON individuais.
