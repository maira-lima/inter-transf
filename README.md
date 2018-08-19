# Projeto de Graduação em Computação UFABC

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/maira-lima/inter-transf/master)

# Gerando Expressões Interação-Transformação com Perceptron de Múltiplas Camadas

## Introdução

A **regressão** é uma técnica estatística que determina a relação entre variáveis através de uma função que aproxima dados obtidos experimentalmente, chamada de modelo. Para tanto, é necessário definir uma função, por exemplo $f(x) = y = ax +b$, a priori e, então, otimizar os parâmetros, nesse exemplo $a$ e $b$, para melhor acomodar os dados, $x$ e $y$.

Uma **rede neural de múltiplas** camadas tem a vantagem de ser um aproximador universal, ou seja, ela é capaz de aproximar qualquer função $f$ com um pequeno erro $\epsilon$, dados os parâmetros corretos. Porém, a função fornecida por essa técnica pode ser composta pela soma de uma grande quantidade de funções não relacionadas, não sendo possível sua interpretação semântica pelo pesquisador. Por esse motivo, essa técnica é chamada de caixa preta.

Quando se tem interesse em um modelo com capacidade de interpretação semântica, utiliza-se a técnica da **regressão simbólica**, que fornece um modelo simples e, ainda assim, de erro mínimo. A regressão simbólica otimiza não apenas os parâmetros, mas também as funções usadas pelo modelo, com os objetivos de minimizar o erro de aproximação e maximizar a simplicidade da expressão.


## Justificativa

Um modelo de regressão simbólica que produz soluções simples, mesmo que com precisão menor que os modelos caixas pretas, pode ser desejável em diversas situações nas quais o objetivo é compreender o sistema, por exemplo fenômenos físicos, ou quando explicar o modelo é necessário para extrair conhecimento do sistema, e, até mesmo, para validar o modelo, como nos seguintes exemplos:

- um modelo que automatiza as decisões de um juiz: não pode ter viés;
- modelo que toma as decisões de um sistema de inteligência artificial que conduz um veículo de passageiros autonomamente;
- muitas empresas online que usam algoritmos para otimizar as ofertas que fazem aos usuários, como a Netflix, a Amazon.

## Representação Interação-Transformação

Dado um conjunto de dados representado por uma matriz $X_{(r \times d)}$, e um vetor de variáveis alvo $y$, a representação Interação-Transformação (IT) foi definida como as funções representáveis pela forma:
\begin{equation}
\hat{y} = \sum_i w_i \cdot g_i(\mathbf{x}),
\label{eq:it}
\end{equation}
onde $w_i$ são coeficientes de uma regressão linear dos componentes $g_i(.)$ que são funções compostas $g(.) = t(.) \circ p(.)$, com $t : \mathbb{R} \rightarrow \mathbb{R}$ uma função de transformação, e $p : \mathbb{R}^d \rightarrow \mathbb{R}$ uma função de interação da forma:
\begin{equation}
p(\mathbf{x}) = \prod_{i=1}^d x_i^{e_i},
\end{equation}
onde $e_i \in \mathbb{Z}$ são expoentes inteiros, denominados força da interação.

## Exemplo

Para exemplificar a estrutura IT, considere a função:
\begin{equation}
f(\mathbf{x}) = 3.5 \cos{(x_1^2 x_2)} + 5 \sqrt{\frac{x_2^3}{x_1}},
\end{equation}
que pode ser descrita como
\begin{align}
\mathbf{t}(z) &= [\cos{z}, \sqrt{z}] \\
\mathbf{p}(\mathbf{x}) &= [x_1^2 x_2, x_1^{-1} x_2^3] \\
\mathbf{w} &= [3.5, 5.] \\
\end{align}
