# FIAP - Global Solution 2026

- **Curso:** Engenharia de Software
    
- **Tema:** Economia Espacial e Análise de Meteoritos
    
- **Ano:** 2026
    

# Sistema de Planejamento de Expedição de Meteoritos

> A plataforma de Análise Espacial é uma aplicação desenvolvida em Python que modela o impacto geográfico de meteoritos utilizando a Teoria dos Grafos. Utilizando dados reais da NASA (`meteorite-landings.csv`), o sistema constrói uma rede de proximidade para a América do Sul e aplica quatro algoritmos fundamentais: Caminho Mínimo (Dijkstra) para roteamento de equipes, Seleção Gulosa para priorização por massa, Programação Dinâmica (Mochila 0/1) para otimização de carga científica, e Simulação Randomizada de falhas climáticas. A implementação prioriza a tipagem estrita, Orientação a Objetos (OOP) via `dataclasses` e tratamento robusto de erros.

## Funcionalidades Principais

A aplicação simula o painel de controle de uma agência espacial, executando um fluxo analítico que permite:

1. **Construção de Grafo Espacial:** Lê e filtra o dataset da NASA, convertendo coordenadas de Latitude/Longitude em nós e arestas baseadas em limites de distância (Fórmula de Haversine).
    
2. **Roteamento de Expedição:** Calcula a rota mais curta e viável entre dois pontos de impacto utilizando o algoritmo de Dijkstra.
    
3. **Coleta de Emergência:** Aplica um Algoritmo Guloso para traçar uma rota rápida priorizando a coleta dos meteoritos de maior massa nas proximidades.
    
4. **Otimização de Carga (Knapsack):** Utiliza Programação Dinâmica para preencher o veículo da expedição maximizando o valor científico (antiguidade da queda) sem exceder o limite de peso.
    
5. **Simulação de Resiliência:** Executa um algoritmo randomizado que destrói rotas probabilisticamente para simular intempéries climáticas (tempestades de areia).
    
6. **Mapeamento Visual:** Renderiza o grafo geográfico em 2D, respeitando o formato continental.
    

## Arquitetura e Estrutura

O projeto é construído em um módulo unificado (`grafo_meteoritos.py`), subdividido em componentes lógicos.

### Classes de Dados (Data Classes)

Estruturas otimizadas para representação de entidades do domínio.

- **`Meteorito`:** Representa o local de queda.
    
    - _Atributos:_ `id_no`, `nome`, `classe_rec`, `massa_g`, `ano`, `lat`, `lon`
        
    - _Métodos Especiais:_ Implementa `__hash__` e `__eq__` para controle eficiente de nós.
        
- **`Aresta`:** Representa a conectividade geográfica.
    
    - _Atributos:_ `id_origem`, `id_destino`, `distancia_km`
        

### Estrutura Central (Modelagem)

- **`GrafoMeteoritos`:** Classe principal que encapsula a lógica matemática e de dados.
    
    - _Atributos:_ `_nos` (Dicionário de nós), `_adjacencia` (Lista de adjacência), `_arestas` (Lista linear de conexões).
        
    - _Métodos:_ `construir_do_dataframe()`, `calcular_caminho_minimo_dijkstra()`, `coleta_gulosa()`, `otimizar_mochila_pd()`, `simular_falha_rotas_randomizado()`, `visualizar_grafo_espacial()`.
        

### Funções Utilitárias (Processamento Auxiliar)

- **`calcular_haversine_km`:** Converte graus de latitude e longitude em distâncias físicas reais (Quilômetros).
    
- **`carregar_e_filtrar_dataset`:** Pipeline de ETL local utilizando `pandas` para limpeza (remoção de valores nulos) e aplicação de _Bounding Box_ geográfico.
    

## Algoritmos Obrigatórios Implementados

- **Caminho Mínimo (Dijkstra):**
    
    - **Propósito:** Encontrar a distância mínima real e a sequência exata de saltos para chegar a um destino.
        
    - **Implementação:** Utiliza a biblioteca nativa `heapq` (Min-Heap) para manter a eficiência de relaxamento das arestas.
        
- **Algoritmo Guloso:**
    
    - **Propósito:** Maximizar a massa coletada com recursos/tempo limitados.
        
    - **Implementação:** Toma a decisão ótima local ordenando os vizinhos disponíveis pela propriedade `massa_g` de forma decrescente a cada passo.
        
- **Programação Dinâmica (Knapsack 0/1):**
    
    - **Propósito:** Resolver o problema de alocação de capacidade do jipe da expedição.
        
    - **Implementação:** Matriz de memoização $O(N \times W)$ que constrói a combinação de itens de maior "valor científico" em relação ao seu peso (massa_g convertida para kg).
        
- **Algoritmo Randomizado:**
    
    - **Propósito:** Simular dinâmica de mundo real onde rotas tornam-se inoperantes.
        
    - **Implementação:** Itera sobre a lista de arestas utilizando `random.random()` avaliando a perda de conectividade contra um limiar de probabilidade estipulado (ex: 15%).
        

## Funcionalidades Técnicas Destacadas

- **Aceleração de Iteração em DataFrames:**
    
    - Substituição do tradicional `.iterrows()` por `.itertuples()` para leitura dos dados brutos, ganhando ordens de grandeza em velocidade.
        
- **Tipagem Estrita (Type Hinting):**
    
    - Uso intensivo de módulos como `typing` (`Dict`, `List`, `Optional`, `Tuple`) garantindo previsibilidade de entrada e saída.
        
    
    Python
    
    ```
    public Tuple[List[int], float] calcular_caminho_minimo_dijkstra(int origem_id, int destino_id)
    ```
    

## Análise de Complexidade (Big O)

|||||
|---|---|---|---|
|**Operação**|**Estrutura/Algoritmo**|**Complexidade**|**Justificativa**|
|Cálculo de Distância|Função `Haversine`|O(1)|Matemática aritmética pura|
|Construção do Grafo|`Loop duplo Aninhado`|O(n²)|Avalia todos contra todos (com n = ~500)|
|Busca de Rota Ótima|Dijkstra (`heapq`)|O((V+E) log V)|Extração do menor caminho na fila de prioridade|
|Rota de Emergência|Guloso|O(k × d log d)|`k` paradas, ordenando `d` vizinhos por massa|
|Seleção de Carga|Mochila Dinâmica|O(n × W)|`n` itens versus capacidade `W` do veículo|
|Destruição de Rotas|Itera Lista Simples|O(E)|Passa linearmente pelo vetor de `E` arestas|

## Como Executar

### Pré-requisitos

- Python 3.9 ou superior
    
- Pip (Gerenciador de pacotes)
    
- Arquivo `meteorite-landings.csv` localizado na raiz ou pasta `/data`.
    

### Execução via Linha de Comando (Prompt/Terminal)

1. Navegue até a pasta raiz do projeto:
    
    Bash
    
    ```
    cd projeto_meteoritos
    ```
    
2. Crie e ative um ambiente virtual (recomendado):
    
    Bash
    
    ```
    python -m venv venv
    # No Windows:
    .\venv\Scripts\activate
    ```
    
3. Instale as bibliotecas necessárias:
    
    Bash
    
    ```
    pip install pandas networkx matplotlib scipy
    ```
    
4. Execute a aplicação:
    
    Bash
    
    ```
    python src/grafo_meteoritos.py
    ```
    

### Execução via IDE (VS Code, PyCharm)

1. Abra a pasta do projeto.
    
2. Certifique-se de selecionar o interpretador Python (`Python: Select Interpreter`) correspondente ao seu ambiente virtual criado.
    
3. Abra o arquivo `grafo_meteoritos.py` e execute utilizando o botão _Run/Play_ no canto superior direito.
    

## Autores

- Arthur Gomes - RM 560771
- Luiz Silva - RM 560110
- Matheus Siroma - RM 560248
- Pedro Estevam - RM 560642
- Witalon Antonio - RM 559023