"""
Módulo  : grafo_meteoritos.py
Propósito: Carregar o dataset de quedas de meteoritos da NASA, filtrar
           para a América do Sul e construir um grafo geográfico de
           proximidade onde os nós são meteoritos e as arestas conectam
           pares a menos de 100 km (peso = distância Haversine).


Dataset : NASA Meteorite Landings (data.nasa.gov / Kaggle)
Região  : América do Sul (lat −55°–13°, lon −82°– −34°)
"""

import json
import logging
import math
import heapq
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GrafoMeteoritos")


# =============================================================================
# Variáveis Constantes
# =============================================================================
CAMINHO_DATASET: str = r"data\meteorite-landings.csv"

# Caixa delimitadora geográfica — América do Sul
LAT_MIN: float = -55.0
LAT_MAX: float = 13.0
LON_MIN: float = -82.0
LON_MAX: float = -34.0

# Limiar máximo de distância para criação de arestas (km)
LIMIAR_DISTANCIA_KM: float = 100.0

# Raio médio da Terra (km) — utilizado na fórmula de Haversine
RAIO_TERRA_KM: float = 6_371.0

# Ano de referência para cálculo do valor científico dos meteoritos
ano_referencia: int = 2025

# Valor científico atribuído a meteoritos com ano de queda desconhecido (NaN)
# Representa uma antiguidade moderadamente alta (equivalente a ~200 anos)
valor_padrao_ano_desconhecido: int = 200


# =============================================================================
# Estruturas de dados principais
# =============================================================================

@dataclass
class Meteorito:
    """
    Objeto de valor imutável que representa um único sítio de queda de
    meteorito. É usado como nó no grafo geográfico.

    Atributos
    ---------
    id_no    : int           — Identificador único do nó no grafo.
    nome     : str           — Nome oficial do meteorito (ex.: 'Achiras').
    classe   : str           — Classificação petrológica (ex.: 'L5', 'H6').
    massa_g  : float         — Massa total em gramas.
    ano      : Optional[int] — Ano de queda ou descoberta (pode ser None).
    lat      : float         — Latitude geodésica em graus (°N positivo).
    lon      : float         — Longitude geodésica em graus (°L positivo).
    """

    id_no   : int
    nome    : str
    classe  : str
    massa_g : float
    ano     : Optional[int]
    lat     : float
    lon     : float

    def __hash__(self) -> int:
        return hash(self.id_no)

    def __eq__(self, outro: object) -> bool:
        if not isinstance(outro, Meteorito):
            return NotImplemented
        return self.id_no == outro.id_no


@dataclass
class Aresta:
    """
    Aresta ponderada e não-direcionada que conecta dois meteoritos.

    Atributos
    ---------
    id_origem    : int   — id_no do nó de partida.
    id_destino   : int   — id_no do nó de chegada.
    distancia_km : float — Peso da aresta: distância Haversine em km.
    """

    id_origem    : int
    id_destino   : int
    distancia_km : float


# =============================================================================
# Funções Utilitárias
# =============================================================================

def haversine_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calcula a distância de círculo máximo em quilômetros entre dois pontos
    sobre a superfície terrestre usando a fórmula de Haversine.

    Parâmetros
    ----------
    lat1, lon1 : float — Coordenadas do primeiro ponto em graus decimais.
    lat2, lon2 : float — Coordenadas do segundo ponto em graus decimais.

    Retorna
    -------
    float — Distância em quilômetros.
    """
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2.0 * RAIO_TERRA_KM * math.asin(math.sqrt(a))


def carregar_e_filtrar_dataset(
    caminho: str = CAMINHO_DATASET,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
) -> pd.DataFrame:
    """
    Carrega o CSV de meteoritos da NASA, remove linhas com coordenadas ou
    massa ausentes e recorta o resultado para a caixa delimitadora geográfica.

    Parâmetros
    ----------
    caminho  : str   — Caminho completo para o arquivo CSV.
    lat_min  : float — Limite sul da caixa delimitadora (inclusivo).
    lat_max  : float — Limite norte da caixa delimitadora (inclusivo).
    lon_min  : float — Limite oeste da caixa delimitadora (inclusivo).
    lon_max  : float — Limite leste da caixa delimitadora (inclusivo).

    Retorna
    -------
    pd.DataFrame — DataFrame filtrado pronto para construção do grafo.

    Lança
    -----
    FileNotFoundError — Se o arquivo CSV não existir no caminho indicado.
    ValueError        — Se o DataFrame resultante for vazio após os filtros.
    """
    caminho_csv = Path(caminho)
    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {caminho}"
        )

    try:
        logger.info("Carregando dataset de '%s' ...", caminho)
        df = pd.read_csv(caminho)
        logger.info(
            "Formato bruto: %d linhas × %d colunas", *df.shape
        )

        # --- Remoção de linhas com campos obrigatórios ausentes -----------
        qtd_antes = len(df)
        df = df.dropna(subset=["reclat", "reclong", "mass"])
        logger.info(
            "Removidas %d linhas com lat/lon/massa ausentes — %d restantes.",
            qtd_antes - len(df),
            len(df),
        )

        # --- Aplicação da caixa delimitadora geográfica -------------------
        mascara_geo = (
            (df["reclat"] >= lat_min) & (df["reclat"] <= lat_max) &
            (df["reclong"] >= lon_min) & (df["reclong"] <= lon_max)
        )
        df = df[mascara_geo].reset_index(drop=True)
        logger.info(
            "Após filtro geográfico (América do Sul): %d meteoritos retidos.",
            len(df),
        )

        if df.empty:
            raise ValueError(
                "Nenhum meteorito restou após o filtro geográfico. "
                "Verifique a caixa delimitadora ou o caminho do dataset."
            )

        return df

    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.error("Falha ao analisar o CSV: %s", exc)
        raise


# =============================================================================
# CLASSE — GrafoMeteoritos
# =============================================================================

class GrafoMeteoritos:
    """
    Grafo não-direcionado e ponderado de sítios de queda de meteoritos,
    representado por lista de adjacência.

    Nós     → objetos :class:`Meteorito` (chave: id_no inteiro)
    Arestas → objetos :class:`Aresta` conectando pares dentro do limiar_km
    Peso    → distância Haversine em quilômetros

    Algoritmos implementados
    ─────────────────────────────────────────────────────────────────
    1. Construção do grafo   construir_de_dataframe()            O(n²)
    2. Caminho mínimo        calcular_caminho_minimo_dijkstra()  O((V+E)·log V)
    3. Rota gulosa           coleta_gulosa()                     O(k · grau_médio)
    4. Mochila 0/1 (PD)     otimizar_mochila_pd()               O(n · W)
    5. Falha randomizada     simular_falha_rotas_randomizado()   O(E + V)

    Parâmetros
    ----------
    limiar_km : float
        Distância máxima em km para dois meteoritos serem conectados por aresta.
    """

    def __init__(self, limiar_km: float = LIMIAR_DISTANCIA_KM) -> None:
        self.limiar_km: float = limiar_km

        # Dicionário principal: id_no → Meteorito
        self._nos: Dict[int, Meteorito] = {}

        # Lista de adjacência: id_no → lista de Arestas (vizinhos diretos)
        self._adjacencia: Dict[int, List[Aresta]] = {}

        # Lista mestra de arestas canônicas únicas (armazenadas uma vez cada)
        self._arestas: List[Aresta] = []

        logger.info(
            "GrafoMeteoritos inicializado — limiar de aresta: %.1f km.",
            limiar_km,
        )

    # ==================================================
    # Propiedades Públicas
    # ==================================================

    @property
    def total_nos(self) -> int:
        """Quantidade total de nós (sítios de meteoritos) no grafo."""
        return len(self._nos)

    @property
    def total_arestas(self) -> int:
        """Quantidade total de arestas únicas não-direcionadas no grafo."""
        return len(self._arestas)

    @property
    def nos(self) -> Dict[int, Meteorito]:
        """Dicionário completo id_no → Meteorito."""
        return self._nos

    @property
    def arestas(self) -> List[Aresta]:
        """Lista mestra de arestas canônicas."""
        return self._arestas

    def vizinhos(self, id_no: int) -> List[Aresta]:
        """Retorna a lista de arestas adjacentes ao nó *id_no*."""
        return self._adjacencia.get(id_no, [])

    # ===============================
    # Auxiliares de Construção
    # ===============================
    def visualizar_grafo_topologico(self) -> None:
        """
        Gera uma visualização clássica de Grafo (Layout de Força/Mola).
        Remove eixos cartesianos e foca estritamente na topologia da rede
        (nós conectados por arestas), eliminando a aparência de dispersão.
        """

        logger.info("Gerando visualização topológica do grafo...")
        grafo_nx = nx.Graph()

        # Filtrar para plotar APENAS nós que possuem conexões.
        nos_conectados = set()
        for aresta in self._arestas:
            nos_conectados.add(aresta.id_origem)
            nos_conectados.add(aresta.id_destino)
            # Adiciona a aresta no NetworkX
            grafo_nx.add_edge(aresta.id_origem, aresta.id_destino, weight=aresta.distancia_km)

        if not nos_conectados:
            logger.warning("Não há arestas para plotar. Tente aumentar o LIMIAR_DISTANCIA_ARESTA_KM.")
            return

        # Configuração da tela
        plt.figure(figsize=(12, 8))

        # Passo 2: Usar o Spring Layout (Algoritmo de atração/repulsão)
        posicoes = nx.spring_layout(grafo_nx, seed=42, k=0.15)

        # Desenhar com foco total nas arestas
        nx.draw_networkx_nodes(
            grafo_nx, 
            posicoes, 
            node_size=60, 
            node_color='#ff7f0e', # Laranja escuro
            edgecolors='black'
        )
        
        nx.draw_networkx_edges(
            grafo_nx, 
            posicoes, 
            width=2.0, 
            alpha=0.8, 
            edge_color='#1f77b4'
        )

        plt.title("Estrutura Topológica do Grafo de Meteoritos", fontsize=16, fontweight='bold')
        
        plt.axis('off') 
        plt.tight_layout()
        
        logger.info("Abrindo interface gráfica (Grafo Topológico)...")
        plt.show()

    def _adicionar_no(self, meteorito: Meteorito) -> None:
        """Registra um Meteorito como nó e inicializa sua lista de adjacência."""
        self._nos[meteorito.id_no] = meteorito
        self._adjacencia[meteorito.id_no] = []

    def _adicionar_aresta(self, aresta: Aresta) -> None:
        """
        Registra uma aresta não-direcionada:
        — na lista mestra `_arestas` (uma vez);
        — em `_adjacencia[origem]` (aresta canônica);
        — em `_adjacencia[destino]` (aresta espelho — sentido inverso).
        """
        self._arestas.append(aresta)
        self._adjacencia[aresta.id_origem].append(aresta)

        # Espelho: mesma distância, sentido inverso, para travessia bidirecional
        aresta_espelho = Aresta(
            id_origem=aresta.id_destino,
            id_destino=aresta.id_origem,
            distancia_km=aresta.distancia_km,
        )
        self._adjacencia[aresta.id_destino].append(aresta_espelho)

    # ========================================
    # CONSTRUÇÃO DO GRAFO (O(n²))
    # ========================================

    def construir_de_dataframe(self, df: pd.DataFrame) -> None:
        """
        Povoa o grafo a partir de um DataFrame já filtrado de meteoritos.

        1 — Criação dos nós
        ─────────────────────────
        Itera sobre as linhas do DataFrame e cria um objeto :class:`Meteorito`
        para cada uma, registrando-o como nó no grafo.

        2 — Criação das arestas (varredura par a par)
        ────────────────────────────────────────────────────
        Para cada par ordenado (i, j) com i < j, calcula a distância
        Haversine. Se a distância for ≤ limiar_km, cria uma aresta
        ponderada entre os dois nós.

        Complexidade de tempo : O(n²)
        Complexidade de espaço: O(n + E)

        Parâmetros
        ----------
        df : pd.DataFrame — DataFrame limpo retornado por
             :func:`carregar_e_filtrar_dataset`.
        """
        logger.info("1 - Criando nós do grafo ...")
        lista_meteoritos: List[Meteorito] = []

        for idx, linha in df.iterrows():
            try:
                ano_bruto = linha.get("year", None)
                m = Meteorito(
                    id_no=int(idx),
                    nome=str(linha["name"]),
                    classe=str(linha.get("recclass", "Desconhecida")),
                    massa_g=float(linha["mass"]),
                    ano=int(ano_bruto) if pd.notna(ano_bruto) else None,
                    lat=float(linha["reclat"]),
                    lon=float(linha["reclong"]),
                )
                self._adicionar_no(m)
                lista_meteoritos.append(m)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Linha %d ignorada — %s", idx, exc)

        logger.info("Nós criados com sucesso: %d.", self.total_nos)
        logger.info(
            "2 - Calculando distâncias par a par (limiar = %.1f km) ...",
            self.limiar_km,
        )

        n = len(lista_meteoritos)
        total_pares_avaliados = 0

        for i in range(n):
            no_a = lista_meteoritos[i]
            for j in range(i + 1, n):
                no_b = lista_meteoritos[j]
                dist = haversine_km(no_a.lat, no_a.lon, no_b.lat, no_b.lon)
                total_pares_avaliados += 1
                if dist <= self.limiar_km:
                    self._adicionar_aresta(
                        Aresta(
                            id_origem=no_a.id_no,
                            id_destino=no_b.id_no,
                            distancia_km=round(dist, 4),
                        )
                    )

        logger.info(
            "Fase 2 concluída: %d pares avaliados — %d arestas criadas "
            "(densidade: %.2f%%).",
            total_pares_avaliados,
            self.total_arestas,
            100.0 * self.total_arestas / max(total_pares_avaliados, 1),
        )

    # ==========================================
    # DIJKSTRA — CAMINHO MÍNIMO (O((V+E)·log V))
    # ==========================================

    def calcular_caminho_minimo_dijkstra(
        self,
        origem_id: int,
        destino_id: int,
    ) -> Tuple[List[int], float]:
        """
        Encontra o caminho de menor distância total (em km) entre dois
        sítios de meteoritos, implementando o algoritmo de Dijkstra do zero.

        Funcionamento
        -------------
        1. Inicializa a distância de todos os nós como infinito, exceto a
           origem (distância zero).
        2. Usa uma fila de prioridade (min-heap) para processar sempre o nó
           com menor distância acumulada conhecida.
        3. Para cada nó processado, relaxa todas as suas arestas: se a
           distância pelo nó atual for menor que a conhecida para o vizinho,
           atualiza e insere na fila.
        4. Encerra assim que o nó destino for retirado da fila (garantia de
           caminho mínimo) ou a fila esgotar (destino inalcançável).
        5. Reconstrói o caminho por backtracking no vetor de predecessores.

        Complexidade de tempo : O((V + E) · log V)
        Complexidade de espaço: O(V)

        Parâmetros
        ----------
        origem_id  : int — id_no do meteorito de partida.
        destino_id : int — id_no do meteorito de chegada.

        Retorna
        -------
        Tuple[List[int], float]
            (caminho, distancia_total_km).
            Caminho é a lista de id_nos do início ao fim.
            Retorna ([], float('inf')) se o destino for inalcançável.

        Lança
        -----
        ValueError — Se origem_id ou destino_id não existirem no grafo.
        """
        # --- Validação dos nós de entrada --------------------------------
        if origem_id not in self._nos:
            raise ValueError(
                f"Nó de origem id={origem_id} não encontrado no grafo."
            )
        if destino_id not in self._nos:
            raise ValueError(
                f"Nó de destino id={destino_id} não encontrado no grafo."
            )

        # Caso especial: origem e destino são o mesmo nó
        if origem_id == destino_id:
            logger.info(
                "[Dijkstra] Origem e destino são o mesmo nó (id=%d).",
                origem_id,
            )
            return ([origem_id], 0.0)

        nome_origem  = self._nos[origem_id].nome
        nome_destino = self._nos[destino_id].nome

        logger.info(
            "[Dijkstra] Buscando caminho mínimo: '%s' (id=%d) → '%s' (id=%d).",
            nome_origem, origem_id, nome_destino, destino_id,
        )

        # --- Inicialização das estruturas do algoritmo -------------------
        # distancias[v] = menor distância acumulada conhecida de origem até v
        distancias: Dict[int, float] = {
            id_no: float("inf") for id_no in self._nos
        }
        distancias[origem_id] = 0.0

        # anteriores[v] = id_no do predecessor no caminho mínimo até v
        anteriores: Dict[int, Optional[int]] = {
            id_no: None for id_no in self._nos
        }

        # Fila de prioridade (min-heap): tupla (distância_acumulada, id_no)
        fila_prioridade: List[Tuple[float, int]] = [(0.0, origem_id)]

        # Conjunto de nós já finalizados (processados definitivamente)
        visitados: set = set()

        logger.info(
            "[Dijkstra] Estruturas inicializadas — %d nós, %d arestas no grafo.",
            self.total_nos, self.total_arestas,
        )

        # --- Loop principal de Dijkstra ----------------------------------
        nos_processados = 0

        while fila_prioridade:
            dist_atual, id_atual = heapq.heappop(fila_prioridade)

            # Entrada obsoleta da fila (nó já finalizado) — ignorar
            if id_atual in visitados:
                continue

            visitados.add(id_atual)
            nos_processados += 1

            # Destino alcançado → encerramento antecipado garantido
            if id_atual == destino_id:
                logger.info(
                    "[Dijkstra] Destino '%s' alcançado! Nós processados: %d.",
                    nome_destino, nos_processados,
                )
                break

            # Relaxamento das arestas dos vizinhos diretos
            for aresta in self._adjacencia[id_atual]:
                id_vizinho = aresta.id_destino

                if id_vizinho in visitados:
                    continue  # Vizinho já finalizado — pular

                nova_distancia = dist_atual + aresta.distancia_km

                # Atualiza somente se encontrou caminho mais curto
                if nova_distancia < distancias[id_vizinho]:
                    distancias[id_vizinho] = nova_distancia
                    anteriores[id_vizinho] = id_atual
                    heapq.heappush(fila_prioridade, (nova_distancia, id_vizinho))

        # --- Verificação de alcançabilidade ------------------------------
        if distancias[destino_id] == float("inf"):
            logger.warning(
                "[Dijkstra] AVISO: Nó de destino '%s' (id=%d) é INALCANÇÁVEL "
                "a partir de '%s' (id=%d). O nó pode estar isolado ou em "
                "componente desconectado.",
                nome_destino, destino_id, nome_origem, origem_id,
            )
            return ([], float("inf"))

        # --- Reconstrução do caminho mínimo por backtracking --------------
        caminho: List[int] = []
        no_atual: Optional[int] = destino_id

        while no_atual is not None:
            caminho.append(no_atual)
            no_atual = anteriores[no_atual]

        caminho.reverse()  # Inverte de [destino → origem] para [origem → destino]
        distancia_total = round(distancias[destino_id], 4)

        logger.info(
            "[Dijkstra] Caminho mínimo encontrado: %d saltos, %.4f km totais.",
            len(caminho) - 1, distancia_total,
        )
        logger.info(
            "[Dijkstra] Rota: %s",
            " → ".join(self._nos[n].nome for n in caminho),
        )

        return (caminho, distancia_total)

    # =====================================================
    # GULOSO — COLETA POR MAIOR MASSA (O(k · grau_médio))
    # =====================================================

    def coleta_gulosa(
        self,
        origem_id: int,
        max_paradas: int = 5,
    ) -> List[int]:
        """
        Planeja uma rota de coleta usando estratégia gulosa: a cada passo,
        o algoritmo escolhe o vizinho direto NÃO visitado com a MAIOR massa
        (massa_g), priorizando espécimes de maior valor bruto para a expedição.

        Lógica de decisão gulosa
        ─────────────────────────
        • Critério de escolha local: max(massa_g) entre vizinhos não visitados.
        • Não há retrocesso: uma vez escolhido o próximo nó, a decisão é final.
        • Encerra se: o limite de paradas for atingido OU não houver mais
          vizinhos não visitados acessíveis a partir do nó atual.

        Obs.: Esta abordagem garante a coleta da maior massa possível a cada
        passo, mas não é globalmente ótima — é uma heurística eficiente.

        Complexidade de tempo : O(max_paradas × grau_médio_do_grafo)

        Parâmetros
        ----------
        origem_id   : int — id_no do meteorito de partida da expedição.
        max_paradas : int — Número máximo de paradas ALÉM da origem (padrão=5).

        Retorna
        -------
        List[int]
            Lista de id_nos na ordem de visita, incluindo a origem.
            Retorna [origem_id] se o nó estiver isolado (sem vizinhos).

        Lança
        -----
        ValueError — Se origem_id não existir no grafo.
        """
        # --- Validação do nó de origem -----------------------------------
        if origem_id not in self._nos:
            raise ValueError(
                f"Nó de origem id={origem_id} não encontrado no grafo."
            )

        meteorito_inicio = self._nos[origem_id]
        logger.info(
            "[Guloso] Iniciando coleta em '%s' (id=%d, massa=%.2f g) "
            "— máx. %d paradas adicionais.",
            meteorito_inicio.nome, origem_id,
            meteorito_inicio.massa_g, max_paradas,
        )

        # --- Tratamento de nó isolado ------------------------------------
        if not self._adjacencia.get(origem_id):
            logger.warning(
                "[Guloso] AVISO: Nó '%s' (id=%d) está ISOLADO — nenhum vizinho "
                "disponível. Retornando rota com apenas a origem.",
                meteorito_inicio.nome, origem_id,
            )
            return [origem_id]

        # --- Inicialização da rota ---------------------------------------
        rota: List[int] = [origem_id]
        visitados: set = {origem_id}
        id_atual: int = origem_id
        massa_total_coletada: float = meteorito_inicio.massa_g

        # --- Loop guloso principal ----------------------------------------
        for passo in range(max_paradas):
            maior_massa_encontrada: float = -1.0
            id_escolhido: Optional[int] = None
            distancia_ate_escolhido: float = 0.0

            # Varre todos os vizinhos do nó atual
            for aresta in self._adjacencia[id_atual]:
                id_candidato = aresta.id_destino

                # Ignora nós já visitados (sem repetição de visitas)
                if id_candidato in visitados:
                    continue

                massa_candidato = self._nos[id_candidato].massa_g

                # Critério guloso: seleciona o vizinho com MAIOR massa
                if massa_candidato > maior_massa_encontrada:
                    maior_massa_encontrada = massa_candidato
                    id_escolhido = id_candidato
                    distancia_ate_escolhido = aresta.distancia_km

            # Nenhum vizinho não-visitado disponível — encerramento antecipado
            if id_escolhido is None:
                logger.warning(
                    "[Guloso] Passo %d: sem vizinhos não-visitados a partir de "
                    "'%s' (id=%d). Rota encerrada com %d paradas no total.",
                    passo + 1,
                    self._nos[id_atual].nome,
                    id_atual,
                    len(rota),
                )
                break

            meteorito_escolhido = self._nos[id_escolhido]
            logger.info(
                "[Guloso] Passo %d/%d: '%s' (id=%d) → '%s' (id=%d) | "
                "massa=%.2f g | dist=%.2f km | decisão: maior massa disponível.",
                passo + 1,
                max_paradas,
                self._nos[id_atual].nome,
                id_atual,
                meteorito_escolhido.nome,
                id_escolhido,
                meteorito_escolhido.massa_g,
                distancia_ate_escolhido,
            )

            # Atualiza estado da expedição
            rota.append(id_escolhido)
            visitados.add(id_escolhido)
            massa_total_coletada += meteorito_escolhido.massa_g
            id_atual = id_escolhido  # Avança para o próximo ponto

        logger.info(
            "[Guloso] Rota concluída: %d paradas realizadas, "
            "%.2f g de massa total coletada.",
            len(rota),
            massa_total_coletada,
        )
        logger.info(
            "[Guloso] Sequência final: %s",
            " → ".join(self._nos[n].nome for n in rota),
        )

        return rota

    # ==============================================
    # MOCHILA 0/1 — PROGRAMAÇÃO DINÂMICA (O(n·W))
    # =============================================

    def otimizar_mochila_pd(
        self,
        capacidade_kg: float,
        lista_ids: List[int],
    ) -> Tuple[List[int], float]:
        """
        Resolve o Problema da Mochila 0/1 por Programação Dinâmica (bottom-up)
        para selecionar o conjunto de meteoritos que maximiza o valor científico
        sem exceder a capacidade de carga do veículo de expedição.

        Definição do valor científico
        ─────────────────────────────
        O valor é inversamente proporcional ao ano de queda — meteoritos mais
        antigos são mais raros e possuem maior importância histórica:

            valor = ano_referencia − ano_de_queda      (para anos conhecidos)
            valor = valor_padrao_ano_desconhecido      (para anos ausentes/NaN)

        Definição do peso
        ─────────────────
        Peso em kg arredondado, com mínimo de 1 kg para evitar peso zero:

            peso_kg = max(1, round(massa_g / 1000))

        Algoritmo DP (bottom-up)
        ─────────────────────────
        tabela_dp[i][w] = maior valor científico obtido usando os primeiros
                          i itens candidatos com capacidade máxima w kg.

        Reconstrução: backtracking da última linha/coluna da tabela DP.

        Complexidade de tempo : O(n × W)  onde W = int(capacidade_kg)
        Complexidade de espaço: O(n × W)

        Parâmetros
        ----------
        capacidade_kg : float     — Capacidade máxima do veículo em kg.
        lista_ids     : List[int] — id_nos dos meteoritos candidatos à seleção.

        Retorna
        -------
        Tuple[List[int], float]
            (ids_selecionados, valor_cientifico_total).
            Retorna ([], 0.0) se nenhum item for elegível.

        Lança
        -----
        ValueError — Se capacidade_kg for zero ou negativa.
        """
        # --- Validação da capacidade -------------------------------------
        if capacidade_kg <= 0:
            raise ValueError(
                f"capacidade_kg deve ser positivo — recebido: {capacidade_kg}."
            )

        capacidade_int: int = int(capacidade_kg)

        logger.info(
            "[Mochila-PD] Iniciando otimização — capacidade: %d kg | "
            "%d meteoritos candidatos.",
            capacidade_int, len(lista_ids),
        )

        # --- Construção da lista de itens elegíveis ----------------------
        # Tupla: (id_no, nome, peso_kg, valor_cientifico, ano)
        itens: List[Tuple[int, str, int, int, Optional[int]]] = []

        for id_no in lista_ids:
            # Verifica existência do nó no grafo
            if id_no not in self._nos:
                logger.warning(
                    "[Mochila-PD] id_no=%d não encontrado no grafo — ignorado.",
                    id_no,
                )
                continue

            m = self._nos[id_no]

            # Converte massa para kg (peso mínimo: 1 kg)
            peso_kg: int = max(1, round(m.massa_g / 1000))

            # Calcula valor científico baseado na antiguidade
            if m.ano is not None:
                valor_cientifico: int = max(0, ano_referencia - m.ano)
            else:
                # Ano desconhecido recebe valor padrão de antiguidade moderada
                valor_cientifico = valor_padrao_ano_desconhecido
                logger.debug(
                    "[Mochila-PD] '%s' sem ano de queda → valor padrão=%d.",
                    m.nome, valor_padrao_ano_desconhecido,
                )

            # Exclui itens que sozinhos já excedem a capacidade total
            if peso_kg > capacidade_int:
                logger.debug(
                    "[Mochila-PD] '%s' (peso=%d kg) excede a capacidade "
                    "(%d kg) — excluído.",
                    m.nome, peso_kg, capacidade_int,
                )
                continue

            itens.append((id_no, m.nome, peso_kg, valor_cientifico, m.ano))

        n_itens = len(itens)
        logger.info(
            "[Mochila-PD] Itens elegíveis para o DP: %d de %d candidatos.",
            n_itens, len(lista_ids),
        )

        if n_itens == 0:
            logger.warning(
                "[Mochila-PD] Nenhum item elegível encontrado. "
                "Verifique os IDs ou aumente a capacidade do veículo."
            )
            return ([], 0.0)

        # --- Preenchimento da tabela DP (bottom-up) ----------------------
        # tabela_dp[i][w] = maior valor com os primeiros i itens, capacidade w
        tabela_dp: List[List[int]] = [
            [0] * (capacidade_int + 1) for _ in range(n_itens + 1)
        ]

        for i in range(1, n_itens + 1):
            _, _, peso_i, valor_i, _ = itens[i - 1]

            for w in range(capacidade_int + 1):
                # Opção 1: não incluir o item i na mochila
                tabela_dp[i][w] = tabela_dp[i - 1][w]

                # Opção 2: incluir o item i, se couber na capacidade w
                if peso_i <= w:
                    valor_com_item_i = tabela_dp[i - 1][w - peso_i] + valor_i
                    if valor_com_item_i > tabela_dp[i][w]:
                        tabela_dp[i][w] = valor_com_item_i

        valor_otimo: int = tabela_dp[n_itens][capacidade_int]
        logger.info(
            "[Mochila-PD] Valor científico ótimo calculado: %d pontos.",
            valor_otimo,
        )

        # --- Backtracking para identificar os itens selecionados ---------
        ids_selecionados: List[int] = []
        capacidade_restante: int = capacidade_int

        for i in range(n_itens, 0, -1):
            # Se o valor mudou ao considerar o item i, ele foi selecionado
            if tabela_dp[i][capacidade_restante] != tabela_dp[i - 1][capacidade_restante]:
                id_no_sel, nome_sel, peso_sel, valor_sel, ano_sel = itens[i - 1]
                ids_selecionados.append(id_no_sel)
                capacidade_restante -= peso_sel
                logger.info(
                    "[Mochila-PD] (OK) Selecionado: '%s' | ano=%s | peso=%d kg | "
                    "valor científico=%d | massa=%.2f g.",
                    nome_sel,
                    str(ano_sel) if ano_sel else "desconhecido",
                    peso_sel,
                    valor_sel,
                    self._nos[id_no_sel].massa_g,
                )

        ids_selecionados.reverse()  # Restaura a ordem de inserção dos itens
        peso_total_utilizado = sum(
            max(1, round(self._nos[i].massa_g / 1000))
            for i in ids_selecionados
        )

        logger.info(
            "[Mochila-PD] Solução final: %d meteoritos selecionados | "
            "%d kg utilizados de %d kg disponíveis | valor total = %d.",
            len(ids_selecionados),
            peso_total_utilizado,
            capacidade_int,
            valor_otimo,
        )

        return (ids_selecionados, float(valor_otimo))

    # ===================================================
    # ANDOMIZADO — SIMULAÇÃO DE FALHA DE ROTAS (O(E + V))
    # ===================================================

    def simular_falha_rotas_randomizado(
        self,
        probabilidade_falha: float = 0.10,
    ) -> int:
        """
        Simula a destruição de rotas por condições climáticas extremas
        (tempestades de areia, inundações, deslizamentos) usando um modelo
        probabilístico aleatório (ensaios de Bernoulli independentes).

        Funcionamento
        ─────────────
        Para cada aresta presente no grafo, gera-se um número aleatório
        uniforme no intervalo [0, 1). Se o valor sorteado for menor que
        `probabilidade_falha`, a aresta é PERMANENTEMENTE removida:
          • da lista mestra `_arestas`;
          • de `_adjacencia[origem]` (sentido canônico);
          • de `_adjacencia[destino]` (sentido espelho).

        O modelo é conservador: arestas removidas não são restauradas.
        Algoritmos subsequentes operarão sobre o grafo degradado.

        Complexidade de tempo : O(E)   — percorre todas as arestas
                              + O(V·grau) — filtra listas de adjacência
        Complexidade de espaço: O(E)   — índices das arestas a remover

        Parâmetros
        ----------
        probabilidade_falha : float
            Probabilidade de cada aresta ser destruída. Deve estar em
            [0.0, 1.0]. Padrão: 0.10 (10% das rotas bloqueadas).

        Retorna
        -------
        int — Número de arestas efetivamente removidas do grafo.

        Lança
        -----
        ValueError — Se probabilidade_falha não estiver no intervalo [0.0, 1.0].
        """
        # --- Validação da probabilidade de falha -------------------------
        if not (0.0 <= probabilidade_falha <= 1.0):
            raise ValueError(
                f"probabilidade_falha deve estar em [0.0, 1.0] — "
                f"recebido: {probabilidade_falha}."
            )

        total_arestas_antes = self.total_arestas
        logger.info(
            "[Randomizado] Simulação de falha de rotas iniciada — "
            "%d arestas no grafo | probabilidade de falha por aresta: %.1f%%.",
            total_arestas_antes,
            probabilidade_falha * 100,
        )

        # --- Sorteio de Bernoulli para cada aresta -----------------------
        # Cada aresta passa por um ensaio independente de probabilidade
        indices_a_remover: set = set()

        for indice, aresta in enumerate(self._arestas):
            numero_sorteado = random.random()  # Uniforme em [0, 1)

            if numero_sorteado < probabilidade_falha:
                # Aresta selecionada para remoção (rota destruída pela tempestade)
                indices_a_remover.add(indice)
                logger.debug(
                    "[Randomizado] Rota destruída: '%s' ↔ '%s' (%.4f km) | "
                    "sorteio=%.4f < limiar=%.4f.",
                    self._nos[aresta.id_origem].nome,
                    self._nos[aresta.id_destino].nome,
                    aresta.distancia_km,
                    numero_sorteado,
                    probabilidade_falha,
                )

        qtd_removidas = len(indices_a_remover)

        # Caso especial: nenhuma aresta foi sorteada para remoção
        if qtd_removidas == 0:
            logger.info(
                "[Randomizado] Nenhuma aresta foi destruída nesta simulação "
                "(todas as rotas sobreviveram à tempestade)."
            )
            return 0

        # --- Constrói conjunto de pares (origem, destino) para filtro ----
        # Necessário para excluir ambas as direções nas listas de adjacência
        pares_destruidos: set = set()
        for indice in indices_a_remover:
            aresta = self._arestas[indice]
            pares_destruidos.add((aresta.id_origem, aresta.id_destino))
            pares_destruidos.add((aresta.id_destino, aresta.id_origem))  # espelho

        # --- Remove das arestas canônicas --------------------------------
        self._arestas = [
            a for idx, a in enumerate(self._arestas)
            if idx not in indices_a_remover
        ]

        # --- Filtra listas de adjacência (ambas as direções) -------------
        for id_no in self._adjacencia:
            self._adjacencia[id_no] = [
                a for a in self._adjacencia[id_no]
                if (a.id_origem, a.id_destino) not in pares_destruidos
            ]

        logger.info(
            "[Randomizado] Simulação concluída: %d de %d arestas destruídas "
            "(taxa real: %.2f%%) | %d arestas sobreviventes.",
            qtd_removidas,
            total_arestas_antes,
            100.0 * qtd_removidas / max(total_arestas_antes, 1),
            self.total_arestas,
        )

        return qtd_removidas

    # =========================================================================
    # Relatório E DIAGNÓSTICOS

    def resumo(self, total_carregados: int) -> dict:
        """
        Retorna um dicionário serializável em JSON com o resumo do grafo.

        Parâmetros
        ----------
        total_carregados : int — Meteoritos carregados do dataset antes da
                                  construção do grafo.
        """
        return {
            "total_meteoritos_carregados": total_carregados,
            "nos_no_grafo": self.total_nos,
            "arestas_no_grafo": self.total_arestas,
            "status": "Grafo gerado com sucesso.",
        }

    def estatisticas_grau(self) -> dict:
        """Retorna grau mínimo, máximo e médio dos nós — útil para diagnóstico."""
        graus = [len(adj) for adj in self._adjacencia.values()]
        if not graus:
            return {}
        return {
            "grau_minimo": min(graus),
            "grau_maximo": max(graus),
            "grau_medio": round(sum(graus) / len(graus), 2),
        }


# ===================
# Pipeline
# ====================

def construir_grafo_meteoritos(
    caminho_dataset: str = CAMINHO_DATASET,
    limiar_km: float = LIMIAR_DISTANCIA_KM,
) -> Tuple[GrafoMeteoritos, int]:
    """
    Pipeline completo: carregar dataset → filtrar → construir grafo.

    Parâmetros
    ----------
    caminho_dataset : str   — Caminho do arquivo CSV do dataset.
    limiar_km       : float — Limiar de distância para criação de arestas.

    Retorna
    -------
    Tuple[GrafoMeteoritos, int]
        O grafo construído e a quantidade de meteoritos carregados.
    """
    df = carregar_e_filtrar_dataset(caminho_dataset)
    total_carregados = len(df)
    grafo = GrafoMeteoritos(limiar_km=limiar_km)
    grafo.construir_de_dataframe(df)
    return grafo, total_carregados


# ============================
# Entrada Principal
# ============================

SEPARADOR = "=" * 70


def main() -> None:
    # Logs de Sistema
    logger.info("Iniciando Carga de Dados e Construção do Grafo...")
    df = carregar_e_filtrar_dataset(CAMINHO_DATASET)
    grafo = GrafoMeteoritos(limiar_km=LIMIAR_DISTANCIA_KM)
    grafo.construir_de_dataframe(df)
    
    if grafo.total_nos == 0:
        logger.warning("Grafo vazio. Encerrando execução.")
        return

    ids_disponiveis = list(grafo._nos.keys())
    origem_teste = ids_disponiveis[0]
    
    # Busca um destino que seja garantidamente alcançável para a demonstração do Dijkstra não dar "infinito"
    vizinhos = grafo._adjacencia.get(origem_teste, [])
    if vizinhos:
        # Pega o destino do vizinho para ter pelo menos 1 salto na demonstração
        destino_teste = vizinhos[-1].id_destino 
    else:
        destino_teste = ids_disponiveis[-1]

    # Execução dos Algortimos
    rota_dijkstra, dist_total = grafo.calcular_caminho_minimo_dijkstra(origem_teste, destino_teste)
    rota_gulosa = grafo.coleta_gulosa(origem_teste, max_paradas=5)
    
    capacidade_veiculo_kg = 50.0 
    amostra_mochila = ids_disponiveis[:20]
    selecionados_pd, valor_pd = grafo.otimizar_mochila_pd(capacidade_veiculo_kg, amostra_mochila)
    
    arestas_destruidas = grafo.simular_falha_rotas_randomizado(probabilidade_falha=0.15)


    # Relatório
    print("\n" + "="*70)
    print(" RELATÓRIO FINAL DA EXPEDIÇÃO: ANÁLISE ESPACIAL ".center(70))
    print("="*70 + "\n")

    print("1. ROTA MAIS RÁPIDA (DIJKSTRA)")
    if dist_total == float('inf'):
         print(f"   Origem {origem_teste} -> Destino {destino_teste} é INALCANÇÁVEL.")
    else:
        caminho_str = " -> ".join(map(str, rota_dijkstra))
        print(f"   Origem: {origem_teste} | Destino: {destino_teste}")
        print(f"   Caminho Mínimo: {caminho_str}")
        print(f"   Distância Total: {dist_total:.2f} km")
    print("-" * 70)

    print("2. COLETA DE EMERGÊNCIA (GULOSO - PRIORIDADE DE MASSA)")
    rota_gul_str = " -> ".join(map(str, rota_gulosa))
    print(f"   Ponto de Partida: {origem_teste}")
    print(f"   Rota Realizada: {rota_gul_str}")
    print("-" * 70)

    print("3. OTIMIZAÇÃO DE CARGA (PROGRAMAÇÃO DINÂMICA - MOCHILA)")
    print(f"   Capacidade do Veículo: {capacidade_veiculo_kg} kg")
    print(f"   Amostra Analisada: {len(amostra_mochila)} meteoritos")
    print(f"   Meteoritos Selecionados (IDs): {selecionados_pd}")
    print(f"   Valor Científico Máximo Obtido: {valor_pd}")
    print("-" * 70)

    print("4. SIMULAÇÃO CLIMÁTICA (RANDOMIZADO)")
    print(f"   Cenário: Tempestade de Areia (Risco de 15% de perda de rota)")
    print(f"   Impacto na Rede: {arestas_destruidas} rotas foram destruídas e estão intransitáveis.")
    print("=" * 70 + "\n")

    # Visualização
    logger.info("Abrindo interface gráfica do mapa de dispersão...")
    grafo.visualizar_grafo_topologico()


if __name__ == "__main__":
    main()
