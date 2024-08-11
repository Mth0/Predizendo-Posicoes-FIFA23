# Resumo

Esse foi um trabalho desenvolvido como projeto final da disciplina "Introdução ao Aprendizado de Máquina" do curso de ciência da computação da UFRJ.

A ideia consiste em experimentar dois modelos distintos com o intuito de prever a posição que dado jogador do FIFA 23 possui. O jogo possui um total de 15 posições e, apesar de 34 atributos numéricos, conseguimos reduzir isso pela metade. Possuímos um notebook e um relatório em PDF. Peço para que se atente que o relatório está desatualizado, visto que houveram mudanças no notebook, que é mais atualizado.

O dataset que utilizamos pode ser encontrado em https://www.kaggle.com/datasets/sanjeetsinghnaik/fifa-23-players-dataset.

## Análise do dataset

O dataset por si só é bem extenso, possuindo mais de 80 colunas e mais de 18000 linhas. Alguns tratamentos foram feitos antes de, de fato, utilizá-los para o treino dos modelos.

### Remoção de duplicatas

Alguns dados eram jogadores duplicados, primeiramente fizemos essa remoção a fim de não utilizar o mesmo dado mais de uma vez durante o treino ou teste dos modelos.

### Filtragem das colunas

Focamos somente nos 34 atributos numéricos de cada jogador como finalização, passe e acelaração. Adicional a isso, utilizamos o atributo "Preferred foot" que diz respeito a qual pé o jogador melhor chuta, cogitamos isso para evitar que o modelo confundisse laterais esquerdos (LB) com laterais direitos (RB), por exemplo. Ocorre que futuramente foi notório que tal atributo não impediu que esse erro ocorresse e talvez pudesse ter sido descartado tranquilamente.

### Remoção de outliers

Fizemos a remoção dos outliers removendo qualquer jogador que possuísse algum atributo cujo z-score fosse maior do que 3 desvios padrões em módulo. Um evento interessante ocorreu após a remoção e pode ser visto no notebook. Mas um outro fato talvez também interessante é que jogadores famosos como Lionel Messi foram removidos do dataset via esse método de remoção.

### Análise da distribuição dos dados

Fizemos uma pequena visualização da distribuição de cada um dos atributos com e sem outliers, não notamos tanta diferença na maioria. Notamos também que muitos se assemelhavam a distribuições normais, isso nos encorajou a futuramente padronizar todos os atributos. De fato, isto não consta no notebook, nem relatório, mas quando ainda utilizavamos os 35 atributos (34 numéricos + "Preferred Foot" transformado) ter padronizado nos deu uma acurácia ~9-10% maior!

![distribuicao_features_sem_outliers](https://github.com/user-attachments/assets/2b9854e7-0622-4620-b610-23bcb76d2a39)
![distribuicao_features_sem_outliers_img2](https://github.com/user-attachments/assets/88c2913b-2a06-417a-9f5a-bdceac41ac5e)


### Análise de correlação entre atributos

34 atributos a princípio parece bastante coisa e será que são todos mesmo necessários? Fizemos uma análise de correlação entre os atributos e nosso atributo target "BestPosition", a posição do jogador, utilizando uma matriz de correlação. Fizemos isso a fim de definir quais atributos seriam importantes e acabou que nenhum atributo se sobresaía muito no quesito alta correlação com o target. Nesse caso, filtramos simplesmente para os que possuíam maior correlação com o target, tomamos cuidado também para não escolher dois atributos que tivessem alta correlação entre si, como atributos de goleiro ou atributos como finalização e voleio, por exemplo. Isso era necessário para evitar informações redudantes que poderiam confundir o modelo. No fim, conseguimos reduzir mais da metade dos atributos sem uma perda significativa de acurácia. No notebook os testes com os 34 atributos não ocorreram, mas no relatório isso pode ser verificado.

![matriz_correlacao](https://github.com/user-attachments/assets/9fdce7af-4185-41c3-a131-eb84e6d3713c)

Alguns padrões interessantes e que, de fato, fazem sentido podem ser vistos como os atributos de goleiro estarem fortemente correlacionados e atributos de ataque como finalização, voleio, força do chute e chute de longe também.

# Modelos

Como comentado, fizemos dois modelos para predições. O primeiro é uma rede neural MLP convencional e o segundo uma combinação de rede neural MLP e floresta aleatória. No 1° modelo para a escolha de qual arquitetura utilizar fizemos um $k$-fold com $k = 5$ e treinamos por 10 épocas em cada fold.

## 1° Modelo: Rede Neural MLP

Aqui utilizamos uma rede neural com input de tamanho reduzido, visto que features foram filtradas anteriormente com a análise de correlação. O atributo "Preferred Foot" foi adicionado também ao input, mas antes foi transformado utilizando uma codificação ordinal. Apesar de não ser exatamente ordenado, utilizar uma codificação ordinal neste caso não impacta tanto, visto que essa feature só possui dois valores possíveis: "Left" e "Right", fazendo-a se comportar como uma variável binária quando ordinalmente codificada. Agora, o output "BestPosition" passou também por uma transformação de categórico para numérico, mas nesta foi utilizada uma codificação One Hot, visto que há 15 possíveis classes. O resultado dessa última transformação é um vetor de tamanho 15, onde cada entrada representa uma classe, deste modo se um jogador pertence à classe 5, a posição 5 receberá o valor $1$ e o restante é zerado. Note que isso resulta em uma matriz bem esparsa, mas não nos aproveitamos dessa estrutura no treino dos modelos.

Após algumas experimentações, a rede neural utilizada continha 14 e 15 como tamanho de input e output, respectivamente, e três camadas escondidas com 32, 64 e 32 neurônios, respectivamente. Adicional a isso, a função de ativação utilizada foi a tangente hiperbólica, o otimizador foi o gradiente descendente estocástico e a função de perda a entropia cruzada categórica.

Treinando por 30 épocas, conseguimos uma acurácia relativamente boa, sendo aproximadamente 74% tanto para treino, quanto para teste. Abaixo as matrizes de confusão

![matriz_confusao_treino_rede_neural](https://github.com/user-attachments/assets/86f9640e-199c-4d21-ba9f-c15cd16eb3d2)
![matriz_confusao_teste_rede_neural](https://github.com/user-attachments/assets/6d77cc86-78ad-4d84-a947-5962018fe295)

Comparativamente com a rede neural do relatório, que utilizava os 35 atributos, considero que esse resultado é aceitável, visto que a acurácia no relatório rondava os 79~80% nos conjuntos de treino e teste. Metade dos atributos foram cortados e obtivemos uma redução somente de 5~6% na acurácia.

### O desbalanceio das classes e posições "redundantes"

Uma coisa bem notória são erros como meias esquerdos (LM) serem erroneamente classificados como meias direitos (RM), onde somente o lado do campo em que o jogador atua muda. Outros erros ocorreram devido a "chutes" do modelo, repare que há menos de 100 segundos atacantes (CF) e mais de 3500 zagueiros (CB) no dataset, portanto dois modos de melhorar este modelo seriam:

- Agrupar as posições redundantes
- Balancear o dataset fazendo, por exemplo, um undersampling.

A segunda estratégia tentamos explorar e isso será melhor discutido posteriormente.

## 2° Modelo: Rede Neural + Floresta Aleatória

Aqui utilizamos uma rede neural de somente uma camada escondida com somente um neurônio, essencialmente uma regressão logística, visto que utilizamos a função de ativação tangente hiperbólica, que pode ser reescrita em função da sigmoide. Um detalhe importante são as classes, todas as 15 foram divididas em dois grupos chamados "ataque" e "defesa". Tal rede neural então era responsável somente por classificar jogadores em uma destas classes. Adicional a isso, duas florestas aleatórias, com 51 árvores cada uma, foram criadas, uma para cada um dos grupos criados anteriormente, isto é, foram treinadas para classificar somente posições, das 15, que foram jogadas em um dos grupos. Desse modo, se um jogador fosse classificado pela rede neural como "ataque", este era redirecionado para a floresta aleatória especializada em posições de "ataque".

Na rede neural a acurácia foi acima de 90% nos conjuntos de treino e teste e a das árvores foram em torno de 74,7% e 77,7% no conjunto de teste para ataque e defesa, respectivamente. Todavia o modelo como um conjunto não atuo tão bem, com uma acurácia de aproximadamente 68,6%. Os problemas encontrados foram parecidos com o do 1° modelo: posições redundantes e desbalanceio de classes. Abaixo a matriz de confusão para todo o dataset.

![matriz_confusao_rede_neural_ensemble](https://github.com/user-attachments/assets/c0ca39b4-ff1b-433a-b86a-171699ebce6f)

## Bônus: Remoção de posições redundantes

Testamos agrupar posições que poderiam confundir o modelo, neste caso criamos dois grupos novos que uniam dois conjuntos de posições:

- SM_W: meias esquerdas e direitas (LM e RM), pontas esquerdas e direitas (LW e RW)
- SB: laterais esquerdos e direitos (LB e RB), alas defensivos esquerdos e direitos (LWB e RWB)

Com isso o número de classes foi reduzido para 9 e somente com essa pequena mudança a acurácia subiu para aproximadamente de 82% e 81% para treino e teste, respectivamente. A matriz de confusão de treino e teste se encontram abaixo.

![matriz_confusao_treino_rede_neural_simplificada](https://github.com/user-attachments/assets/b48065f0-e01c-44dc-9208-c46a9138c996)
![matriz_confusao_teste_rede_neural_simplificada](https://github.com/user-attachments/assets/9205d71b-a5d8-4ce6-844a-e5c0877904ec)

Novamente, no relatório, utilizando todos os 35 atributos, a acurácia era em torno de 86% no treino e teste, uma perda de somente 4~5%!

# Conclusões

Redes neurais pareceu se adequar bem para o problema em questão, apesar de conter alguns problemas que observamos e que certamente cogitamos olhar em um possível futuro. A ideia de undersampling parece promissora e explorar isso soa uma boa ideia. A respeito do segundo modelo, seria interessante entender o porquê da acurácia tão baixa, não ficou muito claro o porquê de isso ter ocorrido. No geral o trabalho foi bem interessante de ser feito e apredemos bastante com essa experiência.

Um último pensamento sobre o agrupamento de posições, o "intuitivo" seria agrupar, de fato, em "ataque" e "defesa", mas essa pode não ser a melhor divisão para o modelo. O uso de um método não-supervisionado como a clusterização poderia ajudar a ver o que, de fato, um modelo poderia considerar similar para, com esse resultado, fazermos as separações de posições no segundo modelo.

# Autoria e agradecimentos

O trabalho foi feito por:

- Matheus do Ó Santos Tiburcio (https://github.com/Mth0)
- Vinícius Lima da Silva Santos (https://github.com/vlimass)

Agradecimentos ao professor João Carlos Pereira da Silva, professor da disciplina, que nos ajudou bastante em algumas tomadas de decisão.
