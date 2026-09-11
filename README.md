# 🎬 Recomendador de Filmes com TensorFlow.js

Projeto criado para praticar conceitos de Machine Learning e Redes Neurais utilizando TensorFlow.js.

A ideia foi construir um recomendador de filmes do zero, utilizando uma base própria de usuários e filmes, treinando o modelo diretamente no navegador e gerando recomendações com base no histórico de cada usuário.

## Como funciona

O projeto foi dividido em duas partes principais: treinamento do modelo e recomendação.

### 1. Treinamento do modelo

Primeiro carregamos os dados dos usuários e dos 100 filmes disponíveis.

Cada filme é transformado em um vetor numérico através do `encodeMovie()`, utilizando:

- ano normalizado
- duração normalizada
- nota normalizada
- gênero em one-hot encoding

Cada filme é representado por um vetor de **14 características**.

Depois, o `encodeUser()` pega todos os filmes assistidos por um usuário, transforma cada um em vetor e calcula a média.

Assim, cada usuário também passa a ser representado por um vetor de 14 posições baseado no seu histórico.

No treinamento combinamos:

```text
14 valores do usuário
+
14 valores do filme
=
28 valores
```

Como temos 6 usuários e 100 filmes:

```text
6 × 100 = 600 exemplos
```

O input utilizado para treinar a rede possui o formato:

```text
[600, 28]
```

Para cada uma dessas 600 combinações existe também um `label`:

```text
1 = usuário assistiu o filme
0 = usuário não assistiu
```

Esses labels funcionam como o gabarito utilizado pela rede durante o treinamento.

## Rede Neural

A rede neural utilizada possui a seguinte estrutura:

```text
28 inputs
↓
Dense 128 - ReLU
↓
Dense 64 - ReLU
↓
Dense 32 - ReLU
↓
Dense 1 - Sigmoid
```

O modelo utiliza:

```text
Optimizer: Adam
Loss: Binary Crossentropy
Metric: Accuracy
```

A última camada utiliza `sigmoid`, retornando um valor entre 0 e 1.

## 2. Recomendação

Depois do modelo treinado, começa a etapa de predição.

Quando um usuário é selecionado, geramos novamente seu vetor médio de 14 posições e combinamos esse vetor com cada um dos 100 filmes.

```text
usuário + filme 1 = 28 valores
usuário + filme 2 = 28 valores
...
usuário + filme 100 = 28 valores
```

O input da predição fica:

```text
[100, 28]
```

Esses dados são enviados para:

```js
model.predict()
```

O modelo retorna um score entre 0 e 1 para cada filme.

Depois disso:

1. removemos os filmes que o usuário já assistiu
2. ordenamos os filmes pelo maior score
3. enviamos as recomendações para o front-end

## Web Worker

O TensorFlow.js roda dentro de um Web Worker, separando o processamento da rede neural da thread principal da interface.

```text
React / Next.js
↓
postMessage()
↓
Web Worker
↓
TensorFlow.js
↓
Treinamento / Predição
↓
postMessage()
↓
React
```

## Tecnologias

- Next.js
- React
- TypeScript
- JavaScript
- TensorFlow.js
- Web Workers
- Tailwind CSS

## Principal aprendizado

O principal aprendizado desse projeto foi entender a separação entre **treinamento** e **predição**.

No treinamento, o modelo recebe:

```text
[600, 28]
```

e aprende utilizando os labels como respostas corretas.

Depois de treinado, para recomendar filmes para um usuário, criamos novos inputs no mesmo formato:

```text
[100, 28]
```

e utilizamos o modelo já treinado para gerar os scores de cada filme.

Ou seja, primeiro criamos e treinamos a inteligência com a nossa base de dados e depois reutilizamos esse modelo para realizar novas predições.