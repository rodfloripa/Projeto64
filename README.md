<p align="justify"><h1>1. Projeto: Graph Contrastive Learning no Zachary Karate Club</h1></p>

<p align="justify">Este projeto implementa um sistema completo de <b>Graph Contrastive Learning (GCL)</b> utilizando o famoso dataset <b>Zachary Karate Club</b>, um dos grafos sociais mais clássicos da literatura de <b>Graph Neural Networks</b>. O objetivo principal é aprender embeddings estruturais de nós sem utilizar rótulos durante o treinamento.</p>

<p align="justify">Em vez de depender de classificação supervisionada, o modelo aprende observando apenas:</p>

<p align="justify">

* conectividade do grafo
* padrões sociais
* estrutura topológica
* relações entre vizinhos

</p>

<p align="justify">O projeto implementa duas abordagens modernas:</p>

<p align="justify">

* <b>GraphCL</b>
* <b>Deep Graph Infomax (DGI)</b>

</p>

<p align="justify">Além disso, o sistema aplica múltiplas técnicas de augmentation em grafos para criar diferentes visões estruturais do mesmo dataset.</p>

---

<p align="justify"><h2>2. O Problema do Aprendizado em Grafos</h2></p>

<p align="justify">Grafos aparecem em inúmeros problemas reais:</p>

<p align="justify">

* redes sociais
* recomendação
* sistemas financeiros
* biologia molecular
* cybersecurity
* telecomunicações
* fraude bancária

</p>

<p align="justify">Diferente de imagens e textos, grafos possuem estrutura não-euclidiana. Cada nó possui quantidade variável de vizinhos, conexões heterogêneas e dependências estruturais complexas. Isso torna extremamente difícil aprender representações eficientes usando modelos tradicionais.</p>

<p align="justify">O aprendizado contrastivo resolve esse problema criando supervisionamento automaticamente a partir do próprio grafo.</p>

---

<p align="justify"><h2>3. Dataset Zachary Karate Club</h2></p>

<p align="justify">O dataset representa relações sociais reais entre membros de um clube de karatê estudado por Wayne Zachary nos anos 1970.</p>

<p align="justify">

* nó → representa uma pessoa
* aresta → representa amizade/interação

</p>

<p align="justify">O clube eventualmente se dividiu em dois grupos reais. Isso torna o dataset excelente para:</p>

<p align="justify">

* clusterização
* detecção de comunidades
* aprendizado não supervisionado
* embeddings estruturais

</p>

<p align="justify">Mesmo sem usar rótulos, embeddings bem aprendidos conseguem naturalmente separar os dois grupos sociais.</p>

---

<p align="justify"><h2>4. Aprendizado Contrastivo</h2></p>

<p align="justify">O aprendizado contrastivo funciona aproximando representações positivas e afastando representações negativas.</p>

<p align="justify">Neste projeto:</p>

<p align="justify">

* duas versões diferentes do mesmo nó → embeddings próximos
* nós estruturalmente diferentes → embeddings distantes

</p>

<p align="justify">O encoder aprende uma função matemática que transforma o grafo em embeddings vetoriais:</p>

<p align="center">

$$
f_\theta : G \rightarrow \mathbb{R}^d
$$

</p>

<p align="justify">Onde:</p>

<p align="justify">

* <b>G</b> representa o grafo
* <b>d</b> representa dimensão do embedding
* <b>θ</b> representa parâmetros treináveis

</p>

<p align="justify">A similaridade entre embeddings é calculada usando similaridade cosseno:</p>

<p align="center">

$$
\text{sim}(z_i,z_j)=\frac{z_i^T z_j}{|z_i||z_j|}
$$

</p>

<p align="justify">A loss contrastiva NT-Xent utilizada no projeto é:</p>

<p align="center">

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

</p>

<p align="justify">Essa função força o modelo a aproximar pares positivos e afastar pares negativos no espaço latente.</p>

---

<p align="justify"><h2>5. Construção das Features Estruturais</h2></p>

<p align="justify">Embora o treinamento seja self-supervised, cada nó precisa possuir atributos iniciais. O projeto constrói features estruturais manualmente:</p>

<p align="justify">

* grau do nó
* clustering coefficient
* pagerank
* betweenness centrality

</p>

<p align="justify">Essas métricas representam propriedades importantes da rede social.</p>

---

<p align="justify"><h3>5.1 Grau do Nó</h3></p>

<p align="justify">O grau mede quantas conexões um nó possui. Nós centrais tendem a possuir alto grau.</p>

<p align="center">

$$
deg(v)=\sum_u A_{vu}
$$

</p>

---

<p align="justify"><h3>5.2 Clustering Coefficient</h3></p>

<p align="justify">Mede quanto os vizinhos de um nó se conectam entre si.</p>

<p align="center">

$$
C_v=
\frac{
2T(v)
}{
deg(v)(deg(v)-1)
}
$$

</p>

---

<p align="justify"><h3>5.3 Pagerank</h3></p>

<p align="justify">Avalia importância estrutural dentro do grafo.</p>

<p align="center">

$$
PR(v)=
\frac{1-d}{N}
+
d
\sum_{u\in M(v)}
\frac{PR(u)}{L(u)}
$$

</p>

---

<p align="justify"><h3>5.4 Betweenness Centrality</h3></p>

<p align="justify">Mede frequência com que um nó aparece em caminhos mínimos.</p>

<p align="center">

$$
BC(v)=
\sum_{s\neq v\neq t}
\frac{\sigma_{st}(v)}{\sigma_{st}}
$$

</p>

---

<p align="justify"><h2>6. Augmentations do Grafo</h2></p>

<p align="justify">Uma das partes mais importantes do projeto é criação de múltiplas versões do mesmo grafo. Isso é equivalente ao augmentation utilizado em visão computacional.</p>

<p align="justify">O projeto utiliza:</p>

<p align="justify">

* edge dropping
* perturbação de features

</p>

<p align="justify">Esses augmentations forçam o modelo a aprender representações robustas.</p>

---

<p align="justify"><h3>6.1 Edge Dropping</h3></p>

<p align="justify">O sistema remove aleatoriamente algumas arestas do grafo. Isso simula conexões ausentes, ruído estrutural e perda parcial de relações sociais.</p>

<p align="justify">O bloco responsável por isso é:</p>

```python
def edge_drop(edge_index, drop_prob=0.2):

    E = edge_index.size(1)

    mask = torch.rand(E) > drop_prob

    return edge_index[:, mask]
```

<p align="justify">A variável <b>E</b> representa número total de arestas. Depois, o sistema gera uma máscara aleatória:</p>

```python
mask = torch.rand(E) > drop_prob
```

<p align="justify">Cada aresta possui probabilidade de remoção igual a:</p>

<p align="center">

$$
P(\text{drop}) = 0.2
$$

</p>

<p align="justify">Por fim:</p>

```python
return edge_index[:, mask]
```

<p align="justify">retorna apenas as arestas preservadas, criando uma nova visão estrutural do grafo.</p>

---

<p align="justify"><h3>6.2 Perturbação de Features</h3></p>

<p align="justify">Além da estrutura, os atributos dos nós também sofrem ruído. Isso evita overfitting estrutural.</p>

```python
def feature_drop(x, drop_prob=0.2):

    mask = torch.rand_like(x) > drop_prob

    return x * mask
```

<p align="justify">A linha abaixo gera máscara aleatória para cada feature:</p>

```python
mask = torch.rand_like(x) > drop_prob
```

<p align="justify">Cada atributo possui probabilidade de remoção igual a:</p>

<p align="center">

$$
P(\text{drop feature}) = 0.2
$$

</p>

<p align="justify">Depois:</p>

```python
return x * mask
```

<p align="justify">zera parcialmente atributos dos nós, forçando robustez dos embeddings.</p>

---

<p align="justify"><h2>7. Encoder GCN</h2></p>

<p align="justify">O projeto utiliza uma <b>Graph Convolutional Network (GCN)</b> como encoder principal.</p>

<p align="justify">A GCN propaga informação entre vizinhos usando a operação:</p>

<p align="center">

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

</p>

<p align="justify">Essa operação permite que cada nó agregue contexto estrutural dos vizinhos.</p>

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

<p align="justify">A primeira camada:</p>

```python
self.conv1 = GCNConv(in_dim, hidden_dim)
```

<p align="justify">realiza agregação estrutural inicial.</p>

<p align="justify">Depois:</p>

```python
x = F.relu(x)
```

<p align="justify">introduz não-linearidade.</p>

<p align="justify">A segunda camada:</p>

```python
self.conv2 = GCNConv(hidden_dim, out_dim)
```

<p align="justify">gera embeddings finais dos nós.</p>

---

<p align="justify"><h2>8. Projection Head</h2></p>

<p align="justify">Após gerar embeddings estruturais, o sistema utiliza uma projection head. Ela melhora estabilidade do aprendizado contrastivo.</p>

```python
self.proj_head = nn.Sequential(
    nn.Linear(EMBED_DIM, EMBED_DIM),
    nn.ReLU(),
    nn.Linear(EMBED_DIM, EMBED_DIM)
)
```

<p align="justify">A projection head cria um espaço intermediário para otimização contrastiva, separando espaço semântico e espaço de treinamento.</p>

---

<p align="justify"><h2>9. Training Loop Contrastivo</h2></p>

<p align="justify">Durante cada época:</p>

<p align="justify">

1. duas versões do grafo são criadas
2. ambas passam pelo encoder
3. embeddings são comparados
4. a loss contrastiva é minimizada

</p>

```python
aug1 = graph_augment(data)
aug2 = graph_augment(data)

_, z1 = model(aug1.x, aug1.edge_index)
_, z2 = model(aug2.x, aug2.edge_index)

loss = contrastive_loss(z1, z2)
```

<p align="justify">As duas views do grafo possuem ruídos diferentes. O modelo aprende embeddings invariantes às perturbações estruturais.</p>

---

<p align="justify"><h2>10. Deep Graph Infomax (DGI)</h2></p>

<p align="justify">Além do GraphCL, o projeto implementa <b>Deep Graph Infomax</b>, que maximiza informação mútua entre embeddings locais e representação global do grafo.</p>

```python
def corruption(self, x):

    idx = torch.randperm(x.size(0))

    return x[idx]
```

<p align="justify">A linha:</p>

```python
idx = torch.randperm(x.size(0))
```

<p align="justify">gera uma permutação aleatória dos nós.</p>

<p align="justify">Depois:</p>

```python
return x[idx]
```

<p align="justify">embaralha atributos, destruindo coerência estrutural.</p>

<p align="justify">O encoder aprende distinguir:</p>

<p align="justify">

* embeddings reais
* embeddings corrompidos

</p>

---

<p align="justify"><h2>11. Visualização dos Embeddings</h2></p>

<p align="justify">Após treinamento, os embeddings são reduzidos para duas dimensões usando TSNE.</p>

<p align="justify">Isso permite visualizar:</p>

<p align="justify">

* comunidades sociais
* clusters
* separação estrutural

</p>

<p align="justify">Mesmo sem utilizar rótulos, os embeddings geralmente formam dois grupos naturais.</p>

---

<p align="justify"><h2>12. Clusterização com KMeans</h2></p>

<p align="justify">Os embeddings aprendidos são avaliados utilizando KMeans.</p>

<p align="justify">A métrica utilizada é NMI:</p>

<p align="center">

$$
NMI(U,V)=
\frac{
2I(U;V)
}{
H(U)+H(V)
}
$$

</p>

<p align="justify">Quanto maior o NMI, melhor qualidade estrutural dos embeddings.</p>

---

<p align="justify"><h2>13. Robustez a Ruído</h2></p>

<p align="justify">O projeto também avalia robustez estrutural adicionando níveis crescentes de ruído:</p>

```python
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
```

<p align="justify">Mesmo com perda parcial de informação, os embeddings continuam preservando estrutura social.</p>

---

<p align="justify"><h2>14. Resultados Obtidos</h2></p>

<p align="justify">O projeto demonstra que Graph Contrastive Learning consegue aprender estrutura social relevante sem utilizar rótulos.</p>

<p align="justify">Os embeddings aprendidos conseguem:</p>

<p align="justify">

* separar comunidades
* preservar relações sociais
* capturar centralidade estrutural
* resistir a ruído
* formar clusters semanticamente coerentes

</p>

<p align="justify">Além disso, o projeto reproduz conceitos modernos utilizados em:</p>

<p align="justify">

* recommendation systems
* social networks
* fraud detection
* molecular learning
* graph foundation models
* cybersecurity

</p>

---

<p align="justify"><h2>15. Possíveis Melhorias</h2></p>

<p align="justify">O projeto pode evoluir para versões ainda mais avançadas:</p>

<p align="justify">

* Graph Attention Networks (GAT)
* hard negative mining
* temporal graphs
* contrastive hierarchical learning
* dynamic graph embeddings
* link prediction
* node classification downstream
* multi-view graph learning

</p>

<p align="justify">Também seria possível utilizar datasets maiores:</p>

<p align="justify">

* Cora
* CiteSeer
* PubMed
* OGB datasets

</p>

---

<p align="justify"><h2>16. Conclusão</h2></p>

<p align="justify">Este projeto implementa um pipeline completo de <b>Graph Contrastive Learning</b> utilizando técnicas modernas de aprendizado self-supervised.</p>

<p align="justify">O sistema demonstra como embeddings estruturais podem emergir naturalmente apenas observando conectividade e relações topológicas.</p>

<p align="justify">Além da implementação prática, o projeto explora conceitos fundamentais da pesquisa moderna em Graph Neural Networks:</p>

<p align="justify">

* contrastive learning
* graph augmentations
* representação auto-supervisionada
* informação mútua
* embeddings estruturais
* robustez topológica

</p>

<p align="justify">O resultado final é um projeto extremamente forte para portfólio em:</p>

<p align="justify">

* Machine Learning
* Deep Learning
* Graph Neural Networks
* Self-Supervised Learning
* AI Research
* Representation Learning

</p>
