<p align="justify">

# 1. Projeto: Graph Contrastive Learning no Zachary Karate Club

</p>

<p align="justify">

## 1.1 Visão Geral

Este projeto implementa um sistema completo de <b>Graph Contrastive Learning (GCL)</b> utilizando o famoso dataset <b>Zachary Karate Club</b>, um dos grafos sociais mais clássicos da literatura de <b>Graph Neural Networks</b>.

O objetivo principal é aprender embeddings estruturais de nós sem utilizar rótulos durante o treinamento.

Em vez de depender de classificação supervisionada, o modelo aprende observando apenas:

* conectividade do grafo
* padrões sociais
* estrutura topológica
* relações entre vizinhos

O projeto implementa duas abordagens modernas:

* <b>GraphCL</b>
* <b>Deep Graph Infomax (DGI)</b>

Além disso, o sistema aplica múltiplas técnicas de augmentation em grafos para criar diferentes visões estruturais do mesmo dataset.

</p>

---

<p align="justify">

# 2. O Problema do Aprendizado em Grafos

</p>

<p align="justify">

Grafos aparecem em inúmeros problemas reais:

* redes sociais
* recomendação
* sistemas financeiros
* biologia molecular
* cybersecurity
* telecomunicações
* fraude bancária

Diferente de imagens e textos, grafos possuem estrutura não-euclidiana.

Cada nó possui:

* número variável de vizinhos
* conexões heterogêneas
* dependências estruturais complexas

Isso torna extremamente difícil aprender representações eficientes usando modelos tradicionais.

O aprendizado contrastivo resolve esse problema criando supervisionamento automaticamente a partir do próprio grafo.

</p>

---

<p align="justify">

# 3. Dataset Zachary Karate Club

</p>

<p align="justify">

O dataset representa relações sociais reais entre membros de um clube de karatê estudado por Wayne Zachary nos anos 1970.

Cada:

* nó → representa uma pessoa
* aresta → representa amizade/interação

O clube eventualmente se dividiu em dois grupos reais.

Isso torna o dataset excelente para:

* clusterização
* detecção de comunidades
* aprendizado não supervisionado
* embeddings estruturais

Mesmo sem usar rótulos, embeddings bem aprendidos conseguem naturalmente separar os dois grupos sociais.

</p>

---

<p align="justify">

# 4. Aprendizado Contrastivo

</p>

<p align="justify">

O aprendizado contrastivo funciona aproximando representações positivas e afastando representações negativas.

Neste projeto:

* duas versões diferentes do mesmo nó → embeddings próximos
* nós estruturalmente diferentes → embeddings distantes

O encoder aprende uma função:

</p>

$$
f_\theta : G \rightarrow \mathbb{R}^d
$$

<p align="justify">

onde:

* (G) representa o grafo
* (d) representa dimensão do embedding
* (\theta) representa parâmetros treináveis

A similaridade entre embeddings é calculada usando similaridade cosseno:

</p>

$$
\text{sim}(z_i,z_j)=\frac{z_i^T z_j}{|z_i||z_j|}
$$

<p align="justify">

A loss contrastiva NT-Xent utilizada no projeto é:

</p>

$$
\mathcal{L}_i =
-\log
\frac{
\exp(\text{sim}(z_i,z_j)/\tau)
}{
\sum_k
\exp(\text{sim}(z_i,z_k)/\tau)
}
$$

<p align="justify">

onde:

* (\tau) representa temperatura
* pares positivos aproximam embeddings
* pares negativos afastam embeddings

</p>

---

<p align="justify">

# 5. Construção das Features Estruturais

</p>

<p align="justify">

Embora o treinamento seja self-supervised, cada nó precisa possuir atributos iniciais.

O projeto constrói features estruturais manualmente:

* grau do nó
* clustering coefficient
* pagerank
* betweenness centrality

Essas métricas representam propriedades importantes da rede social.

</p>

---

<p align="justify">

## 5.1 Grau do Nó

</p>

<p align="justify">

O grau mede quantas conexões um nó possui.

Nós centrais tendem a possuir alto grau.

</p>

$$
deg(v)=\sum_u A_{vu}
$$

---

<p align="justify">

## 5.2 Clustering Coefficient

</p>

<p align="justify">

Mede quanto os vizinhos de um nó se conectam entre si.

</p>

$$
C_v=
\frac{
2T(v)
}{
deg(v)(deg(v)-1)
}
$$

---

<p align="justify">

## 5.3 Pagerank

</p>

<p align="justify">

Avalia importância estrutural dentro do grafo.

</p>

$$
PR(v)=
\frac{1-d}{N}
+
d
\sum_{u\in M(v)}
\frac{PR(u)}{L(u)}
$$

---

<p align="justify">

## 5.4 Betweenness Centrality

</p>

<p align="justify">

Mede frequência com que um nó aparece em caminhos mínimos.

</p>

$$
BC(v)=
\sum_{s\neq v\neq t}
\frac{\sigma_{st}(v)}{\sigma_{st}}
$$

---

<p align="justify">

# 6. Augmentations do Grafo

</p>

<p align="justify">

Uma das partes mais importantes do projeto é criação de múltiplas versões do mesmo grafo.

Isso é equivalente ao augmentation utilizado em visão computacional.

O projeto utiliza:

* edge dropping
* perturbação de features

Esses augmentations forçam o modelo a aprender representações robustas.

</p>

---

<p align="justify">

## 6.1 Edge Dropping

</p>

<p align="justify">

O sistema remove aleatoriamente algumas arestas do grafo.

Isso simula:

* conexões ausentes
* ruído estrutural
* perda parcial de relações sociais

</p>

```python
def edge_drop(edge_index, drop_prob=0.2):

    E = edge_index.size(1)

    mask = torch.rand(E) > drop_prob

    return edge_index[:, mask]
```

<p align="justify">

### Explicação do bloco

</p>

<p align="justify">

A variável:

</p>

```python
E = edge_index.size(1)
```

<p align="justify">

obtém número total de arestas do grafo.

Depois:

</p>

```python
mask = torch.rand(E) > drop_prob
```

<p align="justify">

gera uma máscara aleatória.

Cada aresta possui probabilidade de remoção igual a:

</p>

$$
P(\text{drop}) = 0.2
$$

<p align="justify">

Finalmente:

</p>

```python
return edge_index[:, mask]
```

<p align="justify">

retorna apenas as arestas preservadas.

Isso cria uma nova visão estrutural do grafo.

</p>

---

<p align="justify">

## 6.2 Perturbação de Features

</p>

<p align="justify">

Além da estrutura, os atributos dos nós também sofrem ruído.

Isso evita overfitting estrutural.

</p>

```python
def feature_drop(x, drop_prob=0.2):

    mask = torch.rand_like(x) > drop_prob

    return x * mask
```

<p align="justify">

### Explicação do bloco

</p>

<p align="justify">

A linha:

</p>

```python
mask = torch.rand_like(x) > drop_prob
```

<p align="justify">

gera uma máscara aleatória com mesmo formato das features.

Cada feature possui probabilidade de remoção igual a:

</p>

$$
P(\text{drop feature}) = 0.2
$$

<p align="justify">

Depois:

</p>

```python
return x * mask
```

<p align="justify">

zera parcialmente atributos dos nós.

Isso força o modelo a aprender embeddings robustos mesmo com informação incompleta.

</p>

---

<p align="justify">

# 7. Encoder GCN

</p>

<p align="justify">

O projeto utiliza uma <b>Graph Convolutional Network (GCN)</b> como encoder principal.

A GCN propaga informação entre vizinhos do grafo.

A operação matemática principal é:

</p>

$$
H^{(l+1)}
=========

\sigma
\left(
\hat{D}^{-1/2}
\hat{A}
\hat{D}^{-1/2}
H^{(l)}
W^{(l)}
\right)
$$

<p align="justify">

Essa operação permite que cada nó agregue contexto estrutural dos vizinhos.

</p>

```python
class GCNEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):

        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x
```

---

<p align="justify">

## 7.1 Explicação do Encoder

</p>

<p align="justify">

A primeira camada:

</p>

```python
self.conv1 = GCNConv(in_dim, hidden_dim)
```

<p align="justify">

realiza agregação estrutural inicial.

Ela transforma features originais em representações ocultas.

Depois:

</p>

```python
x = F.relu(x)
```

<p align="justify">

introduz não-linearidade.

Isso permite aprendizado de padrões complexos.

A segunda camada:

</p>

```python
self.conv2 = GCNConv(hidden_dim, out_dim)
```

<p align="justify">

gera embeddings finais dos nós.

Cada embedding passa a conter informação estrutural da vizinhança.

</p>

---

<p align="justify">

# 8. Projection Head

</p>

<p align="justify">

Após gerar embeddings estruturais, o sistema utiliza uma projection head.

Ela melhora estabilidade do aprendizado contrastivo.

</p>

```python
self.proj_head = nn.Sequential(
    nn.Linear(EMBED_DIM, EMBED_DIM),
    nn.ReLU(),
    nn.Linear(EMBED_DIM, EMBED_DIM)
)
```

---

<p align="justify">

## 8.1 Explicação da Projection Head

</p>

<p align="justify">

A projection head cria um espaço intermediário para otimização contrastiva.

Isso separa:

* espaço semântico dos embeddings
* espaço de treinamento contrastivo

A primeira camada linear:

</p>

```python
nn.Linear(EMBED_DIM, EMBED_DIM)
```

<p align="justify">

realiza transformação aprendida dos embeddings.

Depois:

</p>

```python
nn.ReLU()
```

<p align="justify">

introduz não-linearidade.

A última camada produz embeddings finais usados na loss contrastiva.

</p>

---

<p align="justify">

# 9. Training Loop Contrastivo

</p>

<p align="justify">

Durante cada época:

1. duas versões do grafo são criadas
2. ambas passam pelo encoder
3. embeddings são comparados
4. loss contrastiva é minimizada

</p>

```python
aug1 = graph_augment(data)
aug2 = graph_augment(data)

_, z1 = model(aug1.x, aug1.edge_index)
_, z2 = model(aug2.x, aug2.edge_index)

loss = contrastive_loss(z1, z2)
```

---

<p align="justify">

## 9.1 Explicação do Training Loop

</p>

<p align="justify">

A linha:

</p>

```python
aug1 = graph_augment(data)
```

<p align="justify">

gera primeira visão do grafo.

Depois:

</p>

```python
aug2 = graph_augment(data)
```

<p align="justify">

gera segunda visão com ruído diferente.

As duas versões passam pelo modelo:

</p>

```python
_, z1 = model(...)
_, z2 = model(...)
```

<p align="justify">

produzindo embeddings contrastivos.

Por fim:

</p>

```python
loss = contrastive_loss(z1, z2)
```

<p align="justify">

aproxima embeddings positivos e afasta negativos.

</p>

---

<p align="justify">

# 10. Deep Graph Infomax (DGI)

</p>

<p align="justify">

Além do GraphCL, o projeto implementa <b>Deep Graph Infomax</b>.

O DGI maximiza informação mútua entre:

* embeddings locais
* representação global do grafo

O método cria exemplos negativos embaralhando features.

</p>

```python
def corruption(self, x):

    idx = torch.randperm(x.size(0))

    return x[idx]
```

---

<p align="justify">

## 10.1 Explicação do Bloco

</p>

<p align="justify">

A linha:

</p>

```python
idx = torch.randperm(x.size(0))
```

<p align="justify">

gera permutação aleatória dos nós.

Depois:

</p>

```python
return x[idx]
```

<p align="justify">

embaralha atributos.

Isso destrói coerência estrutural.

O encoder aprende distinguir:

* embeddings reais
* embeddings corrompidos

</p>

---

<p align="justify">

# 11. Visualização dos Embeddings

</p>

<p align="justify">

Após treinamento, os embeddings são reduzidos para duas dimensões usando TSNE.

Isso permite visualizar:

* comunidades sociais
* clusters
* separação estrutural

Mesmo sem utilizar rótulos, os embeddings geralmente formam dois grupos naturais.

Isso demonstra que o modelo aprendeu relações sociais latentes.

</p>

---

<p align="justify">

# 12. Clusterização com KMeans

</p>

<p align="justify">

Os embeddings aprendidos são avaliados utilizando KMeans.

Depois, os clusters descobertos são comparados com grupos reais do dataset.

A métrica utilizada é NMI:

</p>

$$
NMI(U,V)=
\frac{
2I(U;V)
}{
H(U)+H(V)
}
$$

<p align="justify">

Quanto maior o NMI:

* melhor qualidade dos embeddings
* melhor separação estrutural
* maior preservação das comunidades

</p>

---

<p align="justify">

# 13. Robustez a Ruído

</p>

<p align="justify">

O projeto também avalia robustez estrutural.

O sistema adiciona níveis crescentes de ruído:

</p>

```python
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
```

<p align="justify">

Mesmo com perda parcial de informação, os embeddings continuam preservando estrutura social.

Isso demonstra robustez do aprendizado contrastivo.

</p>

---

<p align="justify">

# 14. Resultados Obtidos

</p>

<p align="justify">

O projeto demonstra que Graph Contrastive Learning consegue aprender estrutura social relevante sem utilizar rótulos.

Os embeddings aprendidos conseguem:

* separar comunidades
* preservar relações sociais
* capturar centralidade estrutural
* resistir a ruído
* formar clusters semanticamente coerentes

Além disso, o projeto reproduz conceitos modernos utilizados em:

* recommendation systems
* social networks
* fraud detection
* molecular learning
* graph foundation models
* cybersecurity

</p>

---

<p align="justify">

# 15. Possíveis Melhorias

</p>

<p align="justify">

O projeto pode evoluir para versões ainda mais avançadas:

* Graph Attention Networks (GAT)
* hard negative mining
* temporal graphs
* contrastive hierarchical learning
* dynamic graph embeddings
* link prediction
* node classification downstream
* multi-view graph learning

Também seria possível utilizar datasets maiores:

* Cora
* CiteSeer
* PubMed
* OGB datasets

</p>

---

<p align="justify">

# 16. Conclusão

</p>

<p align="justify">

Este projeto implementa um pipeline completo de <b>Graph Contrastive Learning</b> utilizando técnicas modernas de aprendizado self-supervised.

O sistema demonstra como embeddings estruturais podem emergir naturalmente apenas observando conectividade e relações topológicas.

Além da implementação prática, o projeto explora conceitos fundamentais da pesquisa moderna em Graph Neural Networks:

* contrastive learning
* graph augmentations
* representação auto-supervisionada
* informação mútua
* embeddings estruturais
* robustez topológica

O resultado final é um projeto extremamente forte para portfólio em:

* Machine Learning
* Deep Learning
* Graph Neural Networks
* Self-Supervised Learning
* AI Research
* Representation Learning

</p>
