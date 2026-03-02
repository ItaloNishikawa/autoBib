"""
Módulo de Ingestão (Extract).

Responsabilidade: ler todos os arquivos .bib da pasta data/, normalizar os
campos e retornar um DataFrame Pandas limpo e padronizado para as etapas
seguintes do pipeline ETL.
"""

import logging
import re
from pathlib import Path

import bibtexparser
import pandas as pd

logger = logging.getLogger(__name__)

# Ordem padrão das colunas do DataFrame de saída
_COLUMN_ORDER = [
    "base", "indexers", "type", "title", "authors", "year",
    "journal", "volume", "number", "pages", "doi", "url",
    "publisher", "issn", "language", "keywords", "authorKeywords",
    "abstract", "citations_scopus", "note",
]


# ---------------------------------------------------------------------------
# Funções auxiliares de normalização
# ---------------------------------------------------------------------------

def _strip_braces(value: str) -> str:
    """Remove chaves LaTeX residuais de um campo BibTeX.

    Exemplo: ``{Privacy-Preserving Techniques}`` → ``Privacy-Preserving Techniques``
    """
    return re.sub(r"[{}]", "", value).strip()


def _normalize_doi(doi: str) -> str:
    """Padroniza o DOI para minúsculas, garantindo deduplicação consistente."""
    return doi.lower().strip()


def _parse_scopus_citations(note: str) -> int | None:
    """Extrai o número de citações do campo ``note`` exportado pelo Scopus.

    Exemplo de entrada: ``'Cited by: 68; All Open Access; Gold Open Access'``
    """
    match = re.search(r"Cited by:\s*(\d+)", note, re.IGNORECASE)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Parser de entrada individual
# ---------------------------------------------------------------------------

def _parse_entry(entry: dict, source_base: str) -> dict:
    """Converte uma entrada BibTeX bruta em um dicionário padronizado."""

    title  = _strip_braces(entry.get("title", ""))
    doi    = _normalize_doi(entry.get("doi", ""))
    note   = entry.get("note", "")

    # Conference papers usam 'booktitle'; journals usam 'journal'
    journal = entry.get("journal", "") or entry.get("booktitle", "")

    # Converte o ano para inteiro quando possível
    try:
        year: int | str = int(entry.get("year", ""))
    except (ValueError, TypeError):
        year = entry.get("year", "")

    return {
        "base"            : source_base,
        "indexers"        : source_base,
        "type"            : entry.get("ENTRYTYPE", ""),
        "title"           : title,
        "authors"         : entry.get("author", ""),
        "year"            : year,
        "journal"         : journal,
        "volume"          : entry.get("volume", ""),
        "number"          : entry.get("number", ""),
        "pages"           : entry.get("pages", ""),
        "doi"             : doi,
        "url"             : entry.get("url", ""),
        "publisher"       : entry.get("publisher", ""),
        "issn"            : entry.get("issn", ""),
        "language"        : entry.get("language", ""),
        "keywords"        : entry.get("keywords", ""),
        "authorKeywords"  : entry.get("author_keywords", ""),
        "abstract"        : entry.get("abstract", ""),
        "citations_scopus": _parse_scopus_citations(note) if note else None,
        "note"            : note,
    }


# ---------------------------------------------------------------------------
# Função principal do módulo
# ---------------------------------------------------------------------------

def extract_bib_files(folder_path: str = "data/") -> pd.DataFrame:
    """Varre *folder_path*, lê todos os arquivos ``.bib`` e retorna um
    DataFrame padronizado com todos os artigos encontrados.

    Args:
        folder_path: Caminho para a pasta com os arquivos ``.bib``.

    Returns:
        DataFrame com os artigos extraídos, ou DataFrame vazio se nenhum
        arquivo for encontrado ou nenhum artigo for extraído.
    """
    bib_dir   = Path(folder_path)
    bib_files = sorted(bib_dir.glob("*.bib"))

    if not bib_files:
        logger.warning("Nenhum arquivo .bib encontrado em '%s'.", folder_path)
        return pd.DataFrame()

    logger.info("Arquivos .bib encontrados: %s", [f.name for f in bib_files])

    data: list[dict] = []

    for bib_file in bib_files:
        source_base = bib_file.stem.capitalize()
        logger.info("Lendo '%s' (base: %s) ...", bib_file.name, source_base)

        try:
            with bib_file.open("r", encoding="utf-8") as fh:
                bib_database = bibtexparser.load(fh)
        except Exception as exc:
            logger.error(
                "Erro ao ler '%s': %s — arquivo ignorado.", bib_file.name, exc
            )
            continue

        for entry in bib_database.entries:
            data.append(_parse_entry(entry, source_base))

        logger.info("  → %d artigos extraídos.", len(bib_database.entries))

    if not data:
        logger.warning("Nenhum artigo extraído. Verifique os arquivos .bib.")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=_COLUMN_ORDER)
    logger.info("Extração concluída. Total de artigos: %d.", len(df))
    return df


# ---------------------------------------------------------------------------
# Execução direta para testes rápidos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = extract_bib_files()
    if not df.empty:
        print("\nPrévia dos Dados:")
        print(df[["base", "type", "title", "year", "doi", "citations_scopus"]].head(10))
