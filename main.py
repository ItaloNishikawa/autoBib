import logging
import sys

# Deve ser configurado ANTES de importar qualquer módulo que use logging,
# pois basicConfig é no-op se já existir um handler configurado.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from modules.pipeline import PipelineConfig, run

if __name__ == "__main__":

    config = PipelineConfig(
        theme       = "Post-Quantum Cryptography",
        database    = "Scopus",   # Opções: "Scopus", "IEEE", "ACM"
        ai_provider = "groq",     # Opções: "gemini", "groq"
    )

    logger.info(
        "Iniciando Pipeline ETL — tema: '%s' | base: %s | IA: %s",
        config.theme, config.database, config.ai_provider,
    )

    try:
        result = run(config)
        logger.info(
            "Pipeline concluído! %d artigos em '%s'.",
            len(result.final_df), result.zip_path,
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error("Regra de negócio violada: %s", e)
        sys.exit(1)
    except EnvironmentError as e:
        logger.error("Configuração de ambiente: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Falha no pipeline: %s", e)
        sys.exit(1)