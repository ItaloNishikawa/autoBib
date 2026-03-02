import logging

# Deve ser configurado ANTES de importar qualquer módulo que use logging,
# pois basicConfig é no-op se já existir um handler configurado.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from modules import ai_gemini, ai_groq, extractor, processor, exporter

if __name__ == "__main__":

    THEME    = "Post-Quantum Cryptography"
    DATABASE = "Scopus"  # Opções: "Scopus", "IEEE", "ACM"

    # Escolha o provedor de IA para todo o pipeline: "gemini" ou "groq"
    AI_PROVIDER = "groq"

    _ai = ai_gemini if AI_PROVIDER == "gemini" else ai_groq

    logger.info("Iniciando Pipeline ETL — tema: '%s' | base: %s | IA: %s", THEME, DATABASE, AI_PROVIDER)

    # 1. IA gera a query
    # Para economizar cota da API em testes, substitua por strings fixas:
    #   query, justification = "sua query aqui", "sua justificativa aqui"
    query, justification = _ai.generate_query_and_justification(THEME, DATABASE)

    if not query:
        logger.error("Falha ao gerar a query. Verifique a chave de API no .env.")
        raise SystemExit(1)

    # 2. Extract
    raw_df = extractor.extract_bib_files("data")

    if raw_df.empty:
        logger.error("Nenhum artigo extraído. Verifique os arquivos .bib na pasta data/.")
        raise SystemExit(1)

    # 3. Transform
    clean_df = processor.clean_and_deduplicate(raw_df)

    # 4. Enrich (pode demorar — delay por artigo na chamada à IA)
    enriched_df = processor.fetch_citations(clean_df)
    final_df    = _ai.analyze_abstracts(enriched_df)

    # 5. Load / Export
    exporter.export_to_excel(final_df, "output/planilha.xlsx")
    exporter.generate_pdf_report(THEME, query, justification, final_df, "output/relatorio.pdf")
    exporter.create_final_zip("data", "output", "Entrega_Final.zip")

    logger.info("Pipeline concluido! 'Entrega_Final.zip' disponivel em output/.")