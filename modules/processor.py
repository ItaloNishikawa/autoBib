"""
Módulo de Transformação (Transform).

Responsabilidade: garantir a integridade lógica dos dados — deduplicar
artigos por DOI, mesclar indexadores e enriquecer cada registro com o
número de citações atualizado via API do Semantic Scholar.
"""

import logging
import time

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Delay (segundos) entre chamadas à API para evitar rate-limit
_API_DELAY = 0.5
# Timeout (segundos) por requisição HTTP
_REQUEST_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Limpeza e deduplicação
# ---------------------------------------------------------------------------

def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove artigos duplicados por DOI e mescla os indexadores de origem.

    Artigos vindos de múltiplas bases com o mesmo DOI são consolidados em
    uma única linha com ``indexers`` preenchido com todas as origens
    (ex: ``"Scopus, Ieee"``). Artigos sem DOI são mantidos sem alteração.

    Args:
        df: DataFrame bruto retornado por ``extract_bib_files()``.

    Returns:
        DataFrame limpo e deduplicado.
    """
    logger.info("Iniciando limpeza e deduplicação...")

    if df.empty:
        logger.warning("DataFrame vazio. Pulando processamento.")
        return df

    # Trabalha em cópia para não mutar o DataFrame original
    df = df.copy()
    initial_count = len(df)

    # Garante que DOIs vazios sejam NaN (não agrupamos strings vazias juntas)
    df["doi"] = df["doi"].replace("", np.nan)

    # --- Artigos COM DOI ---
    df_with_doi = df.dropna(subset=["doi"]).copy()

    # Monta a coluna 'indexers' mesclando todas as bases que compartilham o mesmo DOI
    # Ex: Scopus + IEEE com mesmo DOI → "Scopus, Ieee"
    merged_indexers = (
        df_with_doi.groupby("doi")["base"]
        .apply(lambda x: ", ".join(sorted(x.unique())))
        .rename("indexers")
    )
    df_with_doi["indexers"] = df_with_doi["doi"].map(merged_indexers)

    # Mantém apenas a primeira ocorrência de cada DOI (já com indexers mesclados)
    df_with_doi = df_with_doi.drop_duplicates(subset=["doi"], keep="first")

    # --- Artigos SEM DOI (mantidos integralmente) ---
    df_without_doi = df[df["doi"].isna()].copy()

    df_clean = pd.concat([df_with_doi, df_without_doi], ignore_index=True)
    final_count = len(df_clean)

    logger.info(
        "Deduplicação concluída: %d originais → %d únicos (%d duplicatas removidas).",
        initial_count, final_count, initial_count - final_count,
    )
    return df_clean


# ---------------------------------------------------------------------------
# Enriquecimento com citações (Semantic Scholar)
# ---------------------------------------------------------------------------

def _get_citations_semantic_scholar(doi: str) -> int | None:
    """Consulta a API pública do Semantic Scholar e retorna a contagem de
    citações para o DOI fornecido.

    Args:
        doi: DOI normalizado (minúsculas) do artigo.

    Returns:
        Número de citações ou ``None`` em caso de falha.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "citationCount"}
    try:
        response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("citationCount")
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.debug("DOI não encontrado no Semantic Scholar: %s", doi)
        else:
            logger.warning("Erro HTTP ao buscar citações para '%s': %s", doi, exc)
    except Exception as exc:
        logger.warning("Erro ao buscar citações para '%s': %s", doi, exc)
    return None


def fetch_citations(df: pd.DataFrame) -> pd.DataFrame:
    """Enriquece o DataFrame com citações obtidas via API do Semantic Scholar.

    Para cada artigo que possui DOI, consulta a API e armazena o resultado
    na coluna ``citations``. Artigos sem DOI recebem ``NaN``.

    Args:
        df: DataFrame (preferencialmente já deduplicado).

    Returns:
        DataFrame com a coluna ``citations`` adicionada.
    """
    if df.empty:
        logger.warning("DataFrame vazio. Pulando busca por citações.")
        return df

    df = df.copy()
    dois_validos = df["doi"].dropna()
    total = len(dois_validos)

    logger.info("Buscando citações para %d artigos com DOI (Semantic Scholar)...", total)

    citations: dict[str, int | None] = {}
    for i, doi in enumerate(dois_validos, start=1):
        citations[doi] = _get_citations_semantic_scholar(doi)
        logger.debug("[%d/%d] DOI: %s → %s citações", i, total, doi, citations[doi])
        time.sleep(_API_DELAY)

    df["citations"] = df["doi"].map(citations)

    found = sum(1 for v in citations.values() if v is not None)
    logger.info(
        "Citações obtidas: %d/%d artigos enriquecidos com sucesso.", found, total
    )
    return df