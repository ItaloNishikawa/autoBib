import logging

from modules import ai_gemini, extractor, processor, exporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    THEME    = "Post-Quantum Cryptography"
    DATABASE = "Scopus"  # Opções: "Scopus", "IEEE", "ACM"

    logger.info("Iniciando Pipeline ETL — tema: '%s' | base: %s", THEME, DATABASE)

    # 1. IA gera a query
    # Para economizar cota da API em testes, substitua por strings fixas:
    #   query, justification = "sua query aqui", "sua justificativa aqui"
    query, justification = ai_gemini.generate_query_and_justification(THEME, DATABASE)

    if not query:
        logger.error("Falha ao gerar a query. Verifique a GEMINI_API_KEY e tente novamente.")
        raise SystemExit(1)

    # 2. Extract
    raw_df = extractor.extract_bib_files("data")

    if raw_df.empty:
        logger.error("Nenhum artigo extraído. Verifique os arquivos .bib na pasta data/.")
        raise SystemExit(1)

    # 3. Transform
    clean_df = processor.clean_and_deduplicate(raw_df)

    # 4. Enrich (pode demorar — ~4s de delay por artigo na chamada à IA)
    enriched_df = processor.fetch_citations(clean_df)
    final_df    = ai_gemini.analyze_abstracts(enriched_df)

    # 5. Load / Export
    exporter.export_to_excel(final_df, "output/planilha.xlsx")
    exporter.generate_pdf_report(THEME, query, justification, final_df, "output/relatorio.pdf")
    exporter.create_final_zip("data", "output", "Entrega_Final.zip")

    logger.info("Pipeline concluido! 'Entrega_Final.zip' disponivel em output/.")