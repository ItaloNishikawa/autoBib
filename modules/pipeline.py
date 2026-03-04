"""
Módulo Orquestrador do Pipeline ETL.

Responsabilidade: centralizar a sequência de etapas do pipeline, desacoplando
a lógica de execução do ponto de entrada (terminal via main.py ou interface
gráfica via app.py). Qualquer etapa pode reportar progresso via callback.
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
# Configuração e Resultado
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Parâmetros de entrada do pipeline, definidos pelo usuário."""
    theme:       str
    database:    str
    ai_provider: str          # "gemini" ou "groq"
    data_dir:    str = "data"
    output_dir:  str = "output"


@dataclass
class PipelineResult:
    """Resultado completo após execução bem-sucedida do pipeline."""
    query:         str              = ""
    justification: str              = ""
    final_df:      pd.DataFrame     = field(default_factory=pd.DataFrame)
    excel_path:    str              = ""
    pdf_path:      str              = ""
    zip_path:      str              = ""


# ---------------------------------------------------------------------------
# Execução do pipeline
# ---------------------------------------------------------------------------

def run(
    config: PipelineConfig,
    on_step: Optional[StepCallback] = None,
) -> PipelineResult:
    """Executa o pipeline ETL completo.

    Args:
        config:  Configuração com tema, base, provedor de IA e diretórios.
        on_step: Callback opcional ``(mensagem, percentual)`` chamado no
                 início de cada etapa — usado pela UI para exibir progresso.
                 Quando ``None`` (modo terminal), apenas o logger é acionado.

    Returns:
        PipelineResult com caminhos dos arquivos gerados e o DataFrame final.

    Raises:
        ValueError:    Violação de regra de negócio (dados insuficientes, etc.).
        RuntimeError:  Falha em etapa crítica (API indisponível, sem artigos, etc.).
        EnvironmentError: Chave de API ausente no .env.
    """

    def _notify(msg: str, pct: int) -> None:
        logger.info(msg)
        if on_step:
            on_step(msg, pct)

    # Valida configuração de entrada antes de qualquer chamada externa
    PipelineRules.validate_config(config)

    result = PipelineResult()
    _ai = ai_gemini if config.ai_provider == "gemini" else ai_groq

    # ── Etapa 1 — Geração de Query ────────────────────────────────────────
    _notify("🤖 Gerando query de busca com IA...", 5)
    result.query, result.justification = _ai.generate_query_and_justification(
        config.theme, config.database
    )
    if not result.query:
        raise RuntimeError(
            "A IA não retornou uma query válida. "
            "Verifique a chave de API no .env e tente novamente."
        )

    # ── Etapa 2 — Extração ────────────────────────────────────────────────
    _notify("📥 Lendo arquivos .bib...", 20)
    raw_df = extractor.extract_bib_files(config.data_dir)
    PipelineRules.validate_extraction(raw_df)

    # ── Etapa 3 — Transformação ───────────────────────────────────────────
    _notify(f"🔄 Limpando e deduplicando {len(raw_df)} artigos...", 40)
    clean_df = processor.clean_and_deduplicate(raw_df)
    PipelineRules.validate_after_transform(clean_df)

    # ── Etapa 4 — Enriquecimento ──────────────────────────────────────────
    _notify(f"🌐 Buscando citações (Semantic Scholar) para {len(clean_df)} artigos...", 55)
    enriched_df = processor.fetch_citations(clean_df)

    _notify("🧠 Analisando abstracts com IA (pode demorar)...", 70)
    PipelineRules.warn_low_abstract_coverage(enriched_df, on_step)
    result.final_df = _ai.analyze_abstracts(enriched_df)

    # ── Etapa 5 — Exportação ──────────────────────────────────────────────
    _notify("💾 Exportando planilha Excel e relatório PDF...", 90)
    result.excel_path = f"{config.output_dir}/planilha.xlsx"
    result.pdf_path   = f"{config.output_dir}/relatorio.pdf"
    result.zip_path   = "output/Entrega_Final.zip"

    exporter.export_to_excel(result.final_df, result.excel_path)
    exporter.generate_pdf_report(
        config.theme,
        result.query,
        result.justification,
        result.final_df,
        result.pdf_path,
    )
    exporter.create_final_zip(config.data_dir, config.output_dir, result.zip_path)

    _notify(f"✅ Pipeline concluído! {len(result.final_df)} artigos processados.", 100)
    return result
