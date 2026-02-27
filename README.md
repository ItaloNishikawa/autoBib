# 📚 AutoBib — Pipeline ETL Acadêmico

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange.svg)

Ferramenta automatizada para extração, processamento, enriquecimento e análise crítica de artigos científicos provenientes de bases de dados acadêmicas (Scopus, IEEE Xplore e ACM Digital Library).

O projeto aplica conceitos de **Engenharia de Dados (ETL)** para transformar arquivos brutos `.bib` em planilhas estruturadas e relatórios comparativos gerados por Inteligência Artificial.

---

## ✨ Funcionalidades

| Etapa | Descrição |
|---|---|
| **Extração (Extract)** | Leitura automatizada de múltiplos arquivos BibTeX (`.bib`), padronizando metadados (Título, Autores, Ano, DOI, etc.). |
| **Transformação (Transform)** | Limpeza de strings e deduplicação inteligente baseada no DOI, mesclando a origem dos indexadores. |
| **Enriquecimento (Enrich)** | Consumo da API do **Semantic Scholar** para citações em tempo real + integração com **Google Gemini** para análise crítica dos *abstracts* (Contribuição, Metodologia, Limitações). |
| **Carga (Load)** | Exportação para planilha Excel (`.xlsx`) formatada e geração de relatório estatístico em PDF. |

---

## 🗂️ Arquitetura do Projeto

A arquitetura foi desenhada com forte separação de responsabilidades *(Separation of Concerns)*:

```text
📁 autoBib/
├── 📁 data/                # Diretório de entrada (arquivos .bib originais)
├── 📁 output/              # Diretório de saída (planilha e PDF gerados)
├── 📄 .env                 # Variáveis de ambiente (chaves de API)
├── 📄 main.py              # Orquestrador principal do pipeline
└── 📁 modules/             # Módulos do sistema
    ├── 📄 __init__.py
    ├── 📄 extractor.py     # Ingestão e parsing de arquivos BibTeX
    ├── 📄 processor.py     # Limpeza, deduplicação e requisições HTTP
    ├── 📄 ai_gemini.py     # Comunicação com a IA (Google Gemini)
    └── 📄 exporter.py      # Geração de arquivos físicos (Excel, PDF, Zip)
```

---

## 🚀 Como Executar

### 1. Pré-requisitos

- Python **3.10** ou superior
- Chave de API do **Google Gemini** — obtenha em [Google AI Studio](https://aistudio.google.com/)

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
pip install pandas bibtexparser openpyxl requests google-generativeai python-dotenv fpdf2
```

### 3. Configuração

Crie um arquivo `.env` na raiz do projeto com sua chave de API:

```env
GEMINI_API_KEY=sua_chave_api_aqui
```

Coloque os arquivos exportados das bases de dados (ex: `scopus.bib`, `ieee.bib`, `acm.bib`) dentro da pasta `data/`.

### 4. Execução

```bash
python main.py
```

Os resultados (planilha Excel e relatório PDF) serão gerados automaticamente na pasta `output/`.

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Uso |
|---|---|
| **Pandas** | Manipulação e análise de dados em memória |
| **BibtexParser** | Interpretação da sintaxe de arquivos `.bib` |
| **Requests** | Consumo de APIs REST (Semantic Scholar) |
| **Google Generative AI SDK** | Integração com LLMs (Gemini) |
| **OpenPyXL** | Geração de planilhas `.xlsx` |
| **FPDF2** | Geração de documentos PDF |
