import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from modules.pipeline import PipelineConfig, generate_queries, run_analysis

if __name__ == "__main__":

    config = PipelineConfig(
        theme       = "Post-Quantum Cryptography",
        ai_provider = "groq",     # Opções: "gemini", "groq"
    )

    # ── Passo 1: gera queries para todas as bases ─────────────────────────
    logger.info("Gerando queries — tema: '%s' | IA: %s", config.theme, config.ai_provider)

    try:
        queries = generate_queries(config)
    except (ValueError, EnvironmentError, RuntimeError) as e:
        logger.error("Falha ao gerar queries: %s", e)
        sys.exit(1)

    print("\n" + "═" * 60)
    for db, (query, justification) in sorted(queries.queries.items()):
        print(f"\n📋 [{db}]\n{query}\n")
        print(f"   Justificativa: {justification}\n")
    print("═" * 60)

    # ── Passo 2: aguarda o usuário colocar os .bib em data/ ───────────────
    input("\nUse as queries acima nas bases de dados, exporte os resultados\n"
          "como .bib e coloque-os na pasta data/.\n\n"
          "Pressione ENTER para iniciar a análise...")

    # ── Passo 3: executa a análise sobre os .bib ──────────────────────────
    logger.info("Iniciando análise dos arquivos .bib...")

    try:
        result = run_analysis(config, queries)
        logger.info(
            "Análise concluída! %d artigos em '%s'.",
            len(result.final_df), result.zip_path,
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error("Regra de negócio violada: %s", e)
        sys.exit(1)
    except EnvironmentError as e:
        logger.error("Configuração de ambiente: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Falha na análise: %s", e)
        sys.exit(1)