# 📚 AutoBib — Pipeline ETL Acadêmico

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange.svg)
![Groq](https://img.shields.io/badge/AI-Groq-darkred.svg)

Ferramenta automatizada para extração, processamento, enriquecimento e análise crítica de artigos científicos provenientes de bases de dados acadêmicas (Scopus, IEEE Xplore e ACM Digital Library).

O projeto aplica conceitos de **Engenharia de Dados (ETL)** para transformar arquivos brutos `.bib` em planilhas estruturadas e relatórios comparativos gerados por Inteligência Artificial.

---

## ✨ Funcionalidades

| Etapa | Descrição |
|---|---|
| **Interface Gráfica** | UI interativa em Streamlit com fluxo guiado em 3 etapas: geração de queries → upload dos `.bib` → resultados. |
| **Geração de Queries** | IA gera automaticamente uma string de busca otimizada para cada base (Scopus, IEEE, ACM), com justificativa técnica. |
| **Extração (Extract)** | Upload direto de arquivos `.bib` pela interface; parsing e padronização de metadados (Título, Autores, Ano, DOI, etc.). |
| **Transformação (Transform)** | Limpeza de strings e deduplicação inteligente baseada no DOI, mesclando a origem dos indexadores. |
| **Enriquecimento (Enrich)** | Consumo da API do **Semantic Scholar** para citações em tempo real + análise crítica dos *abstracts* por IA (Contribuição, Metodologia, Limitações), com feedback artigo a artigo. |
| **Carga (Load)** | Exportação para planilha Excel (`.xlsx`), relatório PDF com as queries selecionadas e pacote ZIP final. |

---

## 🗂️ Arquitetura do Projeto

A arquitetura foi desenhada com forte separação de responsabilidades *(Separation of Concerns)*:

```text
📁 autoBib/
├── 📁 data/                # Diretório de entrada (arquivos .bib — uso via CLI)
├── 📁 output/              # Diretório de saída (planilha, PDF e ZIP gerados)
├── 📄 .env                 # Variáveis de ambiente (chaves de API)
├── 📄 app.py               # Interface gráfica (Streamlit) — ponto de entrada UI
├── 📄 main.py              # Ponto de entrada CLI (terminal)
└── 📁 modules/             # Módulos do sistema
    ├── 📄 __init__.py
    ├── 📄 pipeline.py      # Orquestrador ETL (generate_queries + run_analysis)
    ├── 📄 rules.py         # Regras de negócio e validações centralizadas
    ├── 📄 extractor.py     # Ingestão e parsing de arquivos BibTeX
    ├── 📄 processor.py     # Limpeza, deduplicação e requisições HTTP
    ├── 📄 ai_gemini.py     # Comunicação com a IA (Google Gemini)
    ├── 📄 ai_groq.py       # Comunicação com a IA (Groq)
    └── 📄 exporter.py      # Geração de arquivos físicos (Excel, PDF, Zip)
```

---

## 🚀 Como Executar

### 1. Pré-requisitos

- Python **3.10** ou superior
- Chave de API do **Google Gemini** — obtenha em [Google AI Studio](https://aistudio.google.com/)
- Chave de API do **Groq** (opcional) — obtenha em [console.groq.com](https://console.groq.com/)

### 2. Instalação

Clone o repositório e crie um ambiente virtual:

```bash
git clone https://github.com/seu-usuario/autoBib.git
cd autoBib
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### 3. Configuração

Crie um arquivo `.env` na raiz do projeto com suas chaves de API:

```env
GEMINI_API_KEY=sua_chave_gemini_aqui
GROQ_API_KEY=sua_chave_groq_aqui
```

### 4. Execução

#### Interface Gráfica (recomendado)

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador. O fluxo é guiado em **3 etapas**:

| Etapa | O que acontece |
|---|---|
| **1️⃣ Gerar Queries** | Informe o tema e o provedor de IA. A ferramenta gera uma query otimizada para cada base (Scopus, IEEE, ACM). |
| **2️⃣ Enviar Arquivos** | Use as queries nas bases, exporte os resultados como `.bib` e faça upload direto pela interface. Escolha o provedor de IA para análise e quais queries incluir no PDF. |
| **3️⃣ Resultados** | Visualize os artigos processados e baixe a planilha Excel, o relatório PDF e o pacote ZIP. |

#### Terminal (CLI)

```bash
python main.py
```

Coloque os arquivos `.bib` exportados na pasta `data/` antes de pressionar ENTER. Os resultados serão gerados em `output/`.

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Uso |
|---|---|
| **Streamlit** | Interface gráfica web com fluxo de 3 etapas e feedback em tempo real |
| **Pandas** | Manipulação e análise de dados em memória |
| **NumPy** | Suporte a operações numéricas e vetoriais |
| **BibtexParser** | Interpretação da sintaxe de arquivos `.bib` |
| **Requests** | Consumo de APIs REST (Semantic Scholar) |
| **google-genai** | Integração com LLMs (Google Gemini) |
| **Groq** | Integração com LLMs via Groq Cloud |
| **OpenPyXL** | Geração de planilhas `.xlsx` |
| **FPDF2** | Geração de documentos PDF |
