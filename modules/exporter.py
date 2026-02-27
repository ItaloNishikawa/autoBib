"""
Módulo de Saída (Load).

Responsabilidade: materializar o DataFrame enriquecido nos formatos físicos
exigidos para a entrega — planilha Excel, relatório PDF e pacote ZIP final.
"""

import logging
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
from fpdf import FPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapeamento de colunas internas → nomes exigidos na entrega
# ---------------------------------------------------------------------------
_COLUMN_MAP: dict[str, str] = {
    "base"            : "Base",
    "indexers"        : "Indexadores",
    "type"            : "Tipo",
    "title"           : "Título",
    "authors"         : "Autores",
    "year"            : "Ano",
    "journal"         : "Periódico / Conferência",
    "doi"             : "DOI",
    "keywords"        : "Palavras-chave",
    "authorKeywords"  : "Palavras-chave dos Autores",
    "abstract"        : "Resumo",
    "citations"       : "Citações (Semantic Scholar)",
    "citations_scopus": "Citações (Scopus)",
    "observations"    : "Observações (IA)",
}


# ---------------------------------------------------------------------------
# Auxiliar: sanitiza texto para PDF (Latin-1)
# ---------------------------------------------------------------------------

def _pdf_safe(text: str) -> str:
    """Converte texto Unicode para Latin-1, substituindo caracteres
    incompatíveis pelo equivalente ASCII mais próximo.

    Necessário porque as fontes core do FPDF2 (helvetica, times, etc.)
    não suportam Unicode completo — apenas Latin-1.
    """
    normalized = unicodedata.normalize("NFKD", str(text))
    return normalized.encode("latin-1", errors="ignore").decode("latin-1")


# ---------------------------------------------------------------------------
# Exportação para Excel
# ---------------------------------------------------------------------------

def export_to_excel(
    df: pd.DataFrame,
    output_path: str = "output/planilha.xlsx",
) -> None:
    """Exporta o DataFrame para uma planilha Excel formatada.

    Renomeia as colunas para os nomes exigidos na entrega, mantém apenas
    as colunas relevantes e ajusta automaticamente a largura das colunas.

    Args:
        df: DataFrame processado e enriquecido.
        output_path: Caminho do arquivo ``.xlsx`` de saída.
    """
    logger.info("Gerando planilha Excel...")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Renomeia e filtra somente as colunas presentes no DataFrame
    df_export = df.rename(columns=_COLUMN_MAP)
    colunas_presentes = [v for k, v in _COLUMN_MAP.items() if v in df_export.columns]
    df_export = df_export[colunas_presentes]

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Artigos")

        # Ajusta largura das colunas automaticamente
        ws = writer.sheets["Artigos"]
        for col_cells in ws.columns:
            max_len = max(
                (len(str(cell.value)) for cell in col_cells if cell.value), default=10
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)

    logger.info("Planilha salva em: %s (%d artigos).", out, len(df_export))


# ---------------------------------------------------------------------------
# Análise comparativa (estatísticas)
# ---------------------------------------------------------------------------

def generate_comparative_analysis(df: pd.DataFrame) -> str:
    """Calcula estatísticas comparativas entre as bases e retorna um texto
    formatado pronto para ser inserido no relatório PDF.

    Args:
        df: DataFrame (preferencialmente deduplicado).

    Returns:
        Texto com a análise comparativa.
    """
    logger.info("Calculando estatísticas comparativas...")

    # --- Artigos por base ---
    base_counts = df["base"].value_counts()
    base_top = base_counts.idxmax() if not base_counts.empty else "N/A"

    base_linhas = "\n".join(
        f"  - {base}: {count} artigo(s)" for base, count in base_counts.items()
    )

    # --- Tipos de publicação ---
    tipos = df["type"].str.lower().value_counts()
    n_conf = tipos.get("inproceedings", 0) + tipos.get("conference", 0) + tipos.get("proceedings", 0)
    n_journal = tipos.get("article", 0)

    # --- Artigos indexados em mais de uma base ---
    n_multi = df["indexers"].str.contains(",", na=False).sum()

    # --- Ano de publicações ---
    anos = pd.to_numeric(df["year"], errors="coerce").dropna()
    ano_mais_frequente = int(anos.mode()[0]) if not anos.empty else "N/A"
    intervalo_anos = f"{int(anos.min())}–{int(anos.max())}" if not anos.empty else "N/A"

    analise = (
        f"1. Distribuicao por base de dados:\n{base_linhas}\n"
        f"   -> Base com mais resultados: {base_top} ({base_counts.max()} artigos).\n\n"
        f"2. Tipos de publicacao:\n"
        f"   - Artigos de journal/revista: {n_journal}\n"
        f"   - Artigos de conferencia (proceedings): {n_conf}\n\n"
        f"3. Cobertura temporal: {intervalo_anos} "
        f"(ano mais frequente: {ano_mais_frequente}).\n\n"
        f"4. Artigos indexados em mais de uma base: {n_multi} "
        f"(identificados via DOI e mesclados na coluna 'Indexadores').\n\n"
        f"5. Avaliacao qualitativa: consulte a coluna 'Observacoes (IA)' "
        f"na planilha para a analise critica de cada artigo.\n"
    )
    return analise


# ---------------------------------------------------------------------------
# Geração do PDF
# ---------------------------------------------------------------------------

def generate_pdf_report(
    theme: str,
    query: str,
    justification: str,
    df: pd.DataFrame,
    output_path: str = "output/relatorio.pdf",
) -> None:
    """Gera o relatório final em PDF com query, justificativa e análise comparativa.

    Args:
        theme: Tema da pesquisa.
        query: String de busca gerada pela IA.
        justification: Justificativa técnica da query.
        df: DataFrame final (para calcular as estatísticas).
        output_path: Caminho do arquivo ``.pdf`` de saída.
    """
    logger.info("Gerando relatório PDF...")

    analise = generate_comparative_analysis(df)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.add_page()

    # --- Título ---
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Relatorio Final - Revisao Sistematica", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(3)

    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 8, _pdf_safe(f"Tema: {theme}"), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # --- Seção 1: Query ---
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "1. String de Busca e Justificativa", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, "Query utilizada:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 9)
    pdf.set_fill_color(240, 240, 240)
    pdf.multi_cell(0, 5, _pdf_safe(query), fill=True)
    pdf.ln(3)

    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, "Justificativa tecnica:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 6, _pdf_safe(justification))
    pdf.ln(5)

    # --- Seção 2: Análise Comparativa ---
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "2. Analise Comparativa das Bases", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 6, _pdf_safe(analise))

    pdf.output(str(out))
    logger.info("PDF salvo em: %s.", out)


# ---------------------------------------------------------------------------
# Empacotamento ZIP final
# ---------------------------------------------------------------------------

def create_final_zip(
    data_path: str = "data",
    output_path: str = "output",
    zip_name: str = "Entrega_Final.zip",
) -> None:
    """Compacta a planilha, o PDF e os arquivos ``.bib`` originais em um
    único arquivo ZIP para entrega.

    Args:
        data_path: Pasta com os arquivos ``.bib`` originais.
        output_path: Pasta onde estão a planilha e o PDF gerados.
        zip_name: Nome do arquivo ZIP de saída.
    """
    data_dir  = Path(data_path)
    out_dir   = Path(output_path)
    zip_path  = out_dir / zip_name

    logger.info("Empacotando arquivos em '%s'...", zip_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        # Arquivos .bib originais
        for bib in data_dir.glob("*.bib"):
            zipf.write(bib, arcname=bib.name)
            logger.debug("Adicionado ao ZIP: %s", bib.name)

        # Planilha e PDF gerados
        for artifact in ("planilha.xlsx", "relatorio.pdf"):
            artifact_path = out_dir / artifact
            if artifact_path.exists():
                zipf.write(artifact_path, arcname=artifact)
                logger.debug("Adicionado ao ZIP: %s", artifact)
            else:
                logger.warning("Arquivo não encontrado, ignorado: %s", artifact_path)

    logger.info("ZIP gerado com sucesso: %s.", zip_path)