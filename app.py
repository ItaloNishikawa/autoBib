"""
Interface gráfica do AutoBib — construída com Streamlit.

Execução:
    streamlit run app.py

Fluxo em dois passos:
  Passo 1 — Geração de Queries: usuário informa o tema e a IA gera uma
             query otimizada para cada base (Scopus, IEEE, ACM).
  Passo 2 — Análise: usuário faz upload dos .bib obtidos e o pipeline
             extrai, transforma, enriquece e exporta os resultados.
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from modules.pipeline import PipelineConfig, QueriesResult, generate_queries, run_analysis
from modules.rules import PipelineRules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoBib — Pipeline ETL Acadêmico",
    page_icon="📚",
    layout="centered",
)

st.title("📚 AutoBib")
st.caption("Pipeline ETL Acadêmico — Extração, Transformação e Análise de artigos científicos")

# ---------------------------------------------------------------------------
# Inicialização do estado da sessão
# ---------------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = "config"      # "config" | "queries" | "results"

if "queries_result" not in st.session_state:
    st.session_state.queries_result = None

if "config" not in st.session_state:
    st.session_state.config = None

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None


# ---------------------------------------------------------------------------
# Indicador visual de etapas
# ---------------------------------------------------------------------------
def _render_steps(active: str) -> None:
    steps = {"config": "1️⃣ Gerar Queries", "queries": "2️⃣ Enviar Arquivos", "results": "3️⃣ Resultados"}
    cols = st.columns(len(steps))
    for col, (key, label) in zip(cols, steps.items()):
        if key == active:
            col.markdown(f"**:blue[{label}]**")
        elif list(steps.keys()).index(key) < list(steps.keys()).index(active):
            col.markdown(f"~~{label}~~ ✅")
        else:
            col.markdown(f":gray[{label}]")
    st.divider()


# ===========================================================================
# PASSO 1 — Configuração e geração de queries
# ===========================================================================
if st.session_state.step == "config":
    _render_steps("config")
    st.subheader("⚙️ Configuração")

    with st.form("config_form"):
        theme = st.text_input(
            "🎯 Tema da pesquisa",
            value="Post-Quantum Cryptography",
            help="Descreva o tema em inglês para gerar a melhor query de busca.",
        )
        ai_provider = st.selectbox(
            "🤖 Provedor de IA",
            options=sorted(PipelineRules.VALID_AI_PROVIDERS),
            index=sorted(PipelineRules.VALID_AI_PROVIDERS).index("groq"),
            help="Provedor usado para gerar as queries e analisar os abstracts.",
        )
        st.caption(
            f"A IA gerará automaticamente uma query otimizada para cada base: "
            f"{', '.join(sorted(PipelineRules.VALID_DATABASES))}."
        )
        submitted = st.form_submit_button("🔍 Gerar Queries", width="stretch")

    if submitted:
        try:
            config = PipelineConfig(theme=theme.strip(), ai_provider=ai_provider)
            PipelineRules.validate_config(config)
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()

        st.divider()
        status = st.empty()
        bar    = st.progress(0)

        def _on_step(msg: str, pct: int) -> None:
            if pct >= 0:
                status.info(msg)
                bar.progress(pct)

        try:
            queries = generate_queries(config, on_step=_on_step)
        except EnvironmentError as e:
            st.error(f"🔑 Chave de API ausente: {e}")
            st.info("Crie um arquivo `.env` com `GEMINI_API_KEY` ou `GROQ_API_KEY`.")
            st.stop()
        except RuntimeError as e:
            st.error(f"💥 {e}")
            st.stop()
        except Exception as e:
            st.error(f"💥 Erro inesperado: {type(e).__name__}: {e}")
            st.stop()

        st.session_state.config          = config
        st.session_state.queries_result   = queries
        st.session_state.query_provider   = config.ai_provider
        st.session_state.step             = "queries"
        st.rerun()


# ===========================================================================
# PASSO 2 — Exibição das queries e upload dos .bib
# ===========================================================================
elif st.session_state.step == "queries":
    _render_steps("queries")
    queries: QueriesResult = st.session_state.queries_result
    config: PipelineConfig = st.session_state.config

    st.subheader("🔍 Queries geradas pela IA")
    st.info(
        f"Use as queries abaixo para pesquisar em cada base de dados. "
        f"Após exportar os resultados como `.bib`, faça o upload abaixo."
    )

    for db, (query, justification) in sorted(queries.queries.items()):
        with st.expander(f"📋 {db}", expanded=True):
            st.code(query, language="text")
            if justification:
                st.caption(f"**Justificativa:** {justification}")

    st.divider()
    st.subheader("📂 Upload dos arquivos .bib")

    with st.form("upload_form"):
        uploaded_files = st.file_uploader(
            "Envie os arquivos `.bib` exportados das bases de dados",
            type=["bib"],
            accept_multiple_files=True,
            help=f"Máximo de {PipelineRules.MAX_FILES} arquivos.",
        )
        ai_provider_analysis = st.selectbox(
            "🤖 Provedor de IA para análise",
            options=sorted(PipelineRules.VALID_AI_PROVIDERS),
            index=sorted(PipelineRules.VALID_AI_PROVIDERS).index(config.ai_provider),
            help="Pode ser diferente do provedor usado para gerar as queries.",
        )
        st.markdown("**📄 Queries a incluir no PDF**")
        selected_dbs = {
            db: st.checkbox(db, value=True, key=f"chk_{db}")
            for db in sorted(queries.queries.keys())
        }
        col1, col2 = st.columns(2)
        analyze = col1.form_submit_button("▶️ Analisar Artigos", width="stretch")
        restart = col2.form_submit_button("↩️ Recomeçar",        width="stretch")

    if restart:
        for key in ("step", "queries_result", "config", "pipeline_result"):
            st.session_state.pop(key, None)
        st.rerun()

    if analyze:
        try:
            PipelineRules.validate_uploads(uploaded_files)
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()

        chosen = [db for db, checked in selected_dbs.items() if checked]
        if not chosen:
            st.error("❌ Selecione ao menos uma base para incluir no PDF.")
            st.stop()

        # Filtra somente as queries selecionadas para o PDF
        from modules.pipeline import QueriesResult as _QR
        filtered_queries = _QR(queries={db: queries.queries[db] for db in chosen})

        # Salva os arquivos em pasta temporária e atualiza o provedor de IA
        temp_dir = tempfile.mkdtemp(prefix="autobib_")
        try:
            for uf in uploaded_files:
                (Path(temp_dir) / uf.name).write_bytes(uf.getvalue())
            config.data_dir   = temp_dir
            config.ai_provider = ai_provider_analysis
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            st.error(f"💥 Erro ao salvar arquivos: {e}")
            st.stop()

        st.divider()
        st.subheader("📊 Progresso da análise")
        status      = st.empty()
        bar         = st.progress(0)
        notice_box  = st.empty()   # avisos de cota / erros por artigo

        def _on_analysis_step(msg: str, pct: int) -> None:
            if pct == -1:
                # Erro por artigo — aviso amarelo
                notice_box.warning(msg)
            elif pct == -2:
                # Espera por limite de cota — informativo azul
                notice_box.info(msg)
            else:
                notice_box.empty()   # limpa aviso anterior ao avançar
                status.info(msg)
                bar.progress(pct)

        try:
            result = run_analysis(config, filtered_queries, on_step=_on_analysis_step)
        except (ValueError, FileNotFoundError) as e:
            st.error(f"❌ Regra de negócio violada: {e}")
            st.stop()
        except EnvironmentError as e:
            st.error(f"🔑 Chave de API ausente: {e}")
            st.stop()
        except RuntimeError as e:
            st.error(f"💥 Falha na análise: {e}")
            st.stop()
        except Exception as e:
            st.error(f"💥 Erro inesperado: {type(e).__name__}: {e}")
            st.stop()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        st.session_state.pipeline_result = result
        st.session_state.step = "results"
        st.rerun()


# ===========================================================================
# PASSO 3 — Resultados
# ===========================================================================
elif st.session_state.step == "results":
    _render_steps("results")
    result = st.session_state.pipeline_result
    config: PipelineConfig = st.session_state.config

    st.subheader("✅ Análise concluída!")
    st.success(f"**{len(result.final_df)}** artigos processados.")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("📄 Artigos",      len(result.final_df))
    col_m2.metric("🔍 IA (queries)", st.session_state.get("query_provider", config.ai_provider).capitalize())
    col_m3.metric("🧠 IA (análise)", config.ai_provider.capitalize())

    # Preview do DataFrame
    st.subheader("📋 Prévia dos artigos")
    preview_cols    = ["title", "authors", "year", "journal", "citations", "observations"]
    available_cols  = [c for c in preview_cols if c in result.final_df.columns]
    st.dataframe(
        result.final_df[available_cols] if available_cols else result.final_df,
        width="stretch",
        height=350,
    )

    # Downloads
    st.subheader("📦 Downloads")
    dcol1, dcol2, dcol3 = st.columns(3)

    for col, path, label, fname, mime in [
        (dcol1, result.zip_path,   "📦 Entrega_Final.zip", "Entrega_Final.zip", "application/zip"),
        (dcol2, result.excel_path, "📊 Planilha Excel",    "planilha.xlsx",     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        (dcol3, result.pdf_path,   "📄 Relatório PDF",     "relatorio.pdf",     "application/pdf"),
    ]:
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                col.download_button(label, f, fname, mime, width="stretch")

    st.divider()
    if st.button("↩️ Nova pesquisa", width="stretch"):
        for key in ("step", "queries_result", "config", "pipeline_result"):
            st.session_state.pop(key, None)
        st.rerun()

