"""
Módulo Orquestrador do Pipeline ETL.

Responsabilidade: centralizar a sequência de etapas do pipeline, desacoplando
a lógica de execução do ponto de entrada (terminal via main.py ou interface
gráfica via app.py). Qualquer etapa pode reportar progresso via callback.

Fluxo em duas etapas:
  1. generate_queries()  → gera uma query por base (Scopus, IEEE, ACM)
  2. run_analysis()      → extrai, transforma, enriquece e exporta os .bib
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from modules import ai_gemini, ai_groq, extractor, processor, exporter
from modules.rules import PipelineRules

logger = logging.getLogger(__name__)

# Assinatura do callback de progresso: (mensagem: str, percentual: int 0-100)
StepCallback = Callable[[str, int], None]


# ---------------------------------------------------------------------------
# Configuração e Resultados
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Parâmetros de entrada do pipeline, definidos pelo usuário."""
    theme:       str
    ai_provider: str          # "gemini" ou "groq"
    data_dir:    str = "data"
    output_dir:  str = "output"


@dataclass
class QueriesResult:
    """Queries geradas para cada base de dados.

    Structure: ``{database: (query, justification)}``
    """
    queries: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Resultado completo após execução bem-sucedida da análise."""
    queries:       QueriesResult    = field(default_factory=QueriesResult)
    final_df:      pd.DataFrame     = field(default_factory=pd.DataFrame)
    excel_path:    str              = ""
    pdf_path:      str              = ""
    zip_path:      str              = ""


# ---------------------------------------------------------------------------
# Etapa 1 — Geração de Queries
# ---------------------------------------------------------------------------

def generate_queries(
    config: PipelineConfig,
    on_step: Optional[StepCallback] = None,
) -> QueriesResult:
    """Gera uma query de busca otimizada para cada base de dados suportada.

    Args:
        config:  Configuração com tema e provedor de IA.
        on_step: Callback opcional ``(mensagem, percentual)`` para progresso na UI.

    Returns:
        QueriesResult com as queries indexadas por base de dados.

    Raises:
        RuntimeError:    IA não retornou uma query válida para alguma base.
        EnvironmentError: Chave de API ausente no .env.
    """
    def _notify(msg: str, pct: int) -> None:
        logger.info(msg)
        if on_step:
            on_step(msg, pct)

    PipelineRules.validate_config(config)

    _ai = ai_gemini if config.ai_provider == "gemini" else ai_groq
    databases = sorted(PipelineRules.VALID_DATABASES)
    result = QueriesResult()

    for i, db in enumerate(databases):
        pct = int(10 + (i / len(databases)) * 85)
        _notify(f"🤖 Gerando query para {db}...", pct)

        query, justification = _ai.generate_query_and_justification(config.theme, db)

        if not query:
            raise RuntimeError(
                f"A IA não retornou uma query válida para '{db}'. "
                "Verifique a chave de API no .env e tente novamente."
            )

        result.queries[db] = (query, justification)
        logger.info("Query gerada para %s.", db)

    _notify("✅ Queries geradas para todas as bases!", 100)
    return result


def regenerate_query(
    config: PipelineConfig,
    database: str,
) -> tuple[str, str]:
    """Regenera a query para uma única base de dados.

    Args:
        config:   Configuração com tema e provedor de IA.
        database: Base de dados alvo (ex: ``"Scopus"``).

    Returns:
        Tupla ``(query, justification)`` gerada pela IA.

    Raises:
        RuntimeError:     IA não retornou uma query válida.
        EnvironmentError: Chave de API ausente no .env.
    """
    _ai = ai_gemini if config.ai_provider == "gemini" else ai_groq
    query, justification = _ai.generate_query_and_justification(config.theme, database)
    if not query:
        raise RuntimeError(
            f"A IA não retornou uma query válida para '{database}'. "
            "Verifique a chave de API no .env e tente novamente."
        )
    return query, justification


# ---------------------------------------------------------------------------
# Etapa 2 — Análise dos arquivos .bib
# ---------------------------------------------------------------------------

def run_analysis(
    config: PipelineConfig,
    queries: QueriesResult,
    on_step: Optional[StepCallback] = None,
) -> PipelineResult:
    """Executa as etapas de extração, transformação, enriquecimento e exportação.

    Args:
        config:  Configuração com provedor de IA e diretórios.
        queries: QueriesResult retornado por ``generate_queries()``.
        on_step: Callback opcional ``(mensagem, percentual)`` para progresso na UI.

    Returns:
        PipelineResult com caminhos dos arquivos gerados e o DataFrame final.

    Raises:
        ValueError:      Violação de regra de negócio (dados insuficientes, etc.).
        RuntimeError:    Falha em etapa crítica.
        EnvironmentError: Chave de API ausente no .env.
    """
    def _notify(msg: str, pct: int) -> None:
        logger.info(msg)
        if on_step:
            on_step(msg, pct)

    _ai = ai_gemini if config.ai_provider == "gemini" else ai_groq
    result = PipelineResult(queries=queries)

    # ── Etapa 1 — Extração ────────────────────────────────────────────────
    _notify("📥 Lendo arquivos .bib...", 10)
    raw_df = extractor.extract_bib_files(config.data_dir)
    PipelineRules.validate_extraction(raw_df)

    # ── Etapa 2 — Transformação ───────────────────────────────────────────
    _notify(f"🔄 Limpando e deduplicando {len(raw_df)} artigos...", 30)
    clean_df = processor.clean_and_deduplicate(raw_df)
    PipelineRules.validate_after_transform(clean_df)

    # ── Etapa 3 — Enriquecimento ──────────────────────────────────────────
    _notify(f"🌐 Buscando citações (Semantic Scholar) para {len(clean_df)} artigos...", 50)
    enriched_df = processor.fetch_citations(clean_df)

    _notify("🧠 Analisando abstracts com IA (pode demorar)...", 65)
    PipelineRules.warn_low_abstract_coverage(enriched_df, on_step)

    # Encapsula o callback para mapear 0-100 (interno ao analyze_abstracts)
    # para a faixa 65-90 usada pelo pipeline geral.
    def _abstract_step(msg: str, pct: int) -> None:
        if pct < 0:
            # Avisos e mensagens de espera passam direto (pct -1 ou -2)
            _notify(msg, pct)
        else:
            _notify(msg, int(65 + pct * 0.25))

    result.final_df = _ai.analyze_abstracts(enriched_df, _abstract_step)

    # ── Etapa 4 — Exportação ──────────────────────────────────────────────
    _notify("💾 Exportando planilha Excel e relatório PDF...", 90)
    result.excel_path = f"{config.output_dir}/planilha.xlsx"
    result.pdf_path   = f"{config.output_dir}/relatorio.pdf"
    result.zip_path   = f"{config.output_dir}/Entrega_Final.zip"

    # Formata queries de todas as bases para o PDF
    combined_query = "\n\n".join(
        f"[{db}]\n{q}" for db, (q, _) in queries.queries.items()
    )
    combined_just = "\n\n".join(
        f"[{db}]\n{j}" for db, (_, j) in queries.queries.items()
    )

    exporter.export_to_excel(result.final_df, result.excel_path)
    exporter.generate_pdf_report(
        config.theme,
        combined_query,
        combined_just,
        result.final_df,
        result.pdf_path,
    )
    exporter.create_final_zip(config.data_dir, config.output_dir, "Entrega_Final.zip")

    _notify(f"✅ Análise concluída! {len(result.final_df)} artigos processados.", 100)
    return result

