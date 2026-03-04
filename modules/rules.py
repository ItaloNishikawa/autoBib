"""
Módulo de Regras de Negócio.

Responsabilidade: centralizar todas as validações e restrições do pipeline.
Cada método lança uma exceção tipada específica, permitindo que a UI e o
terminal tratem cada tipo de falha de forma diferente.
"""

import logging
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Assinatura do callback de aviso (reutiliza o tipo do pipeline)
WarnCallback = Callable[[str, int], None]


class PipelineRules:
    """Regras de negócio estáticas aplicadas em cada etapa do pipeline."""

    # ── Limites configuráveis ─────────────────────────────────────────────
    MIN_ARTICLES        = 3     # mínimo de artigos para o pipeline continuar
    MAX_FILES           = 10    # máximo de .bib aceitos por execução
    MAX_ABSTRACT_MISS   = 0.60  # acima deste percentual sem abstract, avisa o usuário
    VALID_DATABASES     = {"Scopus", "IEEE", "ACM"}
    VALID_AI_PROVIDERS  = {"gemini", "groq"}
    # ── Validação de arquivos enviados ──────────────────────────────────────

    @staticmethod
    def validate_uploads(files: list) -> None:
        """Valida os arquivos .bib enviados pelo usuário na interface.

        Raises:
            ValueError: Nenhum arquivo enviado ou limite excedido.
        """
        if not files:
            raise ValueError(
                "Nenhum arquivo .bib enviado. "
                "Adicione ao menos 1 arquivo antes de executar o pipeline."
            )
        if len(files) > PipelineRules.MAX_FILES:
            raise ValueError(
                f"{len(files)} arquivo(s) enviados, mas o limite é {PipelineRules.MAX_FILES}. "
                "Reduza a quantidade de arquivos e tente novamente."
            )
        logger.info("Upload validado: %d arquivo(s) .bib.", len(files))
    # ── Validação de configuração ─────────────────────────────────────────

    @staticmethod
    def validate_config(config) -> None:
        """Valida os parâmetros informados pelo usuário antes de qualquer
        chamada de API ou I/O.

        Raises:
            ValueError: Parâmetro inválido ou ausente.
        """
        if not config.theme or not config.theme.strip():
            raise ValueError(
                "O campo 'Tema da pesquisa' não pode estar vazio."
            )
        if len(config.theme.strip()) < 3:
            raise ValueError(
                "O tema da pesquisa deve ter ao menos 3 caracteres."
            )
        if config.ai_provider not in PipelineRules.VALID_AI_PROVIDERS:
            raise ValueError(
                f"Provedor de IA inválido: '{config.ai_provider}'. "
                f"Use 'gemini' ou 'groq'."
            )

    # ── Validação pós-extração ────────────────────────────────────────────

    @staticmethod
    def validate_extraction(df: pd.DataFrame) -> None:
        """Verifica se a extração retornou dados utilizáveis.

        Raises:
            FileNotFoundError: Nenhum arquivo .bib encontrado na pasta.
            ValueError: Menos artigos do que o mínimo necessário.
        """
        if df is None or df.empty:
            raise FileNotFoundError(
                "Nenhum artigo extraído. "
                "Verifique se os arquivos .bib enviados são válidos e contêm entradas."
            )
        if len(df) < PipelineRules.MIN_ARTICLES:
            raise ValueError(
                f"Apenas {len(df)} artigo(s) encontrado(s). "
                f"O pipeline requer ao menos {PipelineRules.MIN_ARTICLES} para continuar."
            )

        logger.info("Extração validada: %d artigos encontrados.", len(df))

    # ── Validação pós-transformação ───────────────────────────────────────

    @staticmethod
    def validate_after_transform(df: pd.DataFrame) -> None:
        """Verifica se o DataFrame ainda possui artigos após deduplicação.

        Raises:
            ValueError: Todos os artigos foram removidos como duplicatas.
        """
        if df.empty:
            raise ValueError(
                "Nenhum artigo sobrou após deduplicação. "
                "Os arquivos .bib podem conter apenas entradas duplicadas."
            )

        logger.info("Transformação validada: %d artigos únicos.", len(df))

    # ── Aviso de cobertura de abstracts ──────────────────────────────────

    @staticmethod
    def warn_low_abstract_coverage(
        df: pd.DataFrame,
        on_step: Optional[WarnCallback] = None,
    ) -> None:
        """Emite aviso (não bloqueia) quando muitos artigos estão sem abstract.

        Artigos sem abstract reduzem a qualidade da análise de IA. O pipeline
        continua, mas o usuário é informado via callback e logger.

        Args:
            df:      DataFrame após fetch de citações.
            on_step: Callback opcional para exibir o aviso na UI.
        """
        if "abstract" not in df.columns:
            return

        missing = df["abstract"].isna().sum() + (df["abstract"] == "").sum()
        pct = missing / len(df) if len(df) > 0 else 0

        if pct > PipelineRules.MAX_ABSTRACT_MISS:
            msg = (
                f"⚠️  {pct:.0%} dos artigos estão sem abstract — "
                "a análise de IA será parcial. "
                "Considere exportar os arquivos .bib com o campo 'abstract' incluído."
            )
            logger.warning(msg)
            if on_step:
                # Percentual -1 sinaliza aviso (sem avanço de barra)
                on_step(msg, -1)
