"""
Interface gráfica do AutoBib — construída com Streamlit.

Execução:
    streamlit run app.py
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Garante que o diretório raiz do projeto esteja no PATH de importação,
# independente do diretório de trabalho ao iniciar o Streamlit.
sys.path.insert(0, str(Path(__file__).parent))

from modules.pipeline import PipelineConfig, run
from modules.rules import PipelineRules

# ---------------------------------------------------------------------------
# Configuração de logging (redireciona para o terminal onde Streamlit roda)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Layout da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoBib — Pipeline ETL Acadêmico",
    page_icon="📚",
    layout="centered",
)

st.title("📚 AutoBib")
st.caption("Pipeline ETL Acadêmico — Extração, Transformação e Análise de artigos científicos")
st.divider()

# ---------------------------------------------------------------------------
# Painel de configuração
# ---------------------------------------------------------------------------
st.subheader("⚙️ Configuração do Pipeline")

with st.form("pipeline_form"):
    theme = st.text_input(
        "🎯 Tema da pesquisa",
        value="Post-Quantum Cryptography",
        help="Descreva o tema em inglês para gerar a melhor query de busca.",
    )

    col1, col2 = st.columns(2)
    database = col1.selectbox(
        "🗄️ Base de dados",
        options=sorted(PipelineRules.VALID_DATABASES),
        index=sorted(PipelineRules.VALID_DATABASES).index("Scopus"),
        help="Base da qual os arquivos .bib foram exportados.",
    )
    ai_provider = col2.selectbox(
        "🤖 Provedor de IA",
        options=sorted(PipelineRules.VALID_AI_PROVIDERS),
        index=sorted(PipelineRules.VALID_AI_PROVIDERS).index("groq"),
        help="Provedor usado para gerar a query e analisar os abstracts.",
    )

    st.caption(
        "📁 Os arquivos `.bib` devem estar na pasta `data/` do projeto antes de executar."
    )

    submitted = st.form_submit_button("▶️ Executar Pipeline", use_container_width=True)

# ---------------------------------------------------------------------------
# Execução do pipeline
# ---------------------------------------------------------------------------
if submitted:
    st.divider()
    st.subheader("📊 Progresso")

    # Widgets de progresso (criados antes da execução para atualização incremental)
    status_placeholder  = st.empty()
    progress_placeholder = st.empty()
    warning_placeholder  = st.empty()

    progress_placeholder.progress(0)
    status_placeholder.info("🚀 Iniciando pipeline...")

    config = PipelineConfig(
        theme=theme.strip(),
        database=database,
        ai_provider=ai_provider,
    )

    # ── Validação antecipada (sem chamar API) ─────────────────────────────
    try:
        PipelineRules.validate_config(config)
    except ValueError as e:
        status_placeholder.empty()
        progress_placeholder.empty()
        st.error(f"❌ Configuração inválida: {e}")
        st.stop()

    # ── Callback de progresso para o pipeline ─────────────────────────────
    def update_ui(msg: str, pct: int) -> None:
        """Atualiza os widgets de progresso na UI do Streamlit."""
        if pct == -1:
            # Aviso não bloqueante (cobertura de abstracts)
            warning_placeholder.warning(msg)
        else:
            status_placeholder.info(msg)
            progress_placeholder.progress(pct)

    # ── Execução ──────────────────────────────────────────────────────────
    try:
        result = run(config, on_step=update_ui)

    except (ValueError, FileNotFoundError) as e:
        status_placeholder.empty()
        progress_placeholder.empty()
        st.error(f"❌ Regra de negócio violada: {e}")
        st.stop()

    except EnvironmentError as e:
        status_placeholder.empty()
        progress_placeholder.empty()
        st.error(f"🔑 Erro de configuração de ambiente: {e}")
        st.info(
            "Crie um arquivo `.env` na raiz do projeto com as chaves de API:\n\n"
            "```\nGEMINI_API_KEY=...\nGROQ_API_KEY=...\n```"
        )
        st.stop()

    except RuntimeError as e:
        status_placeholder.empty()
        progress_placeholder.empty()
        st.error(f"💥 Falha no pipeline: {e}")
        st.stop()

    except Exception as e:
        status_placeholder.empty()
        progress_placeholder.empty()
        st.error(f"💥 Erro inesperado: {type(e).__name__}: {e}")
        st.stop()

    # ── Resultados ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("✅ Resultados")
    st.success(f"Pipeline concluído! **{len(result.final_df)}** artigos processados.")

    # Métricas rápidas
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("📄 Artigos", len(result.final_df))
    col_m2.metric("🗄️ Base", config.database)
    col_m3.metric("🤖 IA usada", config.ai_provider.capitalize())

    # Query gerada pela IA
    with st.expander("🔍 Query gerada pela IA"):
        st.code(result.query, language="text")
        if result.justification:
            st.caption(result.justification)

    # Preview do DataFrame
    st.subheader("📋 Prévia dos artigos")
    preview_cols = ["title", "authors", "year", "journal", "citations", "observations"]
    available_cols = [c for c in preview_cols if c in result.final_df.columns]
    st.dataframe(
        result.final_df[available_cols] if available_cols else result.final_df,
        use_container_width=True,
        height=350,
    )

    # Botões de download
    st.subheader("📦 Downloads")
    dcol1, dcol2, dcol3 = st.columns(3)

    zip_path   = Path(result.zip_path)
    excel_path = Path(result.excel_path)
    pdf_path   = Path(result.pdf_path)

    if zip_path.exists():
        with open(zip_path, "rb") as f:
            dcol1.download_button(
                label="📦 Entrega_Final.zip",
                data=f,
                file_name="Entrega_Final.zip",
                mime="application/zip",
                use_container_width=True,
            )

    if excel_path.exists():
        with open(excel_path, "rb") as f:
            dcol2.download_button(
                label="📊 Planilha Excel",
                data=f,
                file_name="planilha.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            dcol3.download_button(
                label="📄 Relatório PDF",
                data=f,
                file_name="relatorio.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
