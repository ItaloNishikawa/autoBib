# Mapa da Arquitetura do Projeto

Detalhamento da responsabilidade única de cada arquivo `.py`, seguindo o fluxo **ETL** (Extract → Transform → Load).

---

## 1. `main.py` — O Orquestrador / Maestro

> **Objetivo:** Ponto de entrada do programa. Não executa processamento pesado; apenas dita a ordem dos acontecimentos.

**O que deve conter:**

- Definição de variáveis globais (ex: `THEME`)
- O bloco `if __name__ == "__main__":`
- Chamadas sequenciais para os módulos:
  ```
  extrair → limpar → enriquecer → exportar
  ```

---

## 2. `modules/extractor.py` — Módulo de Ingestão / Extract

> **Objetivo:** Ler dados não estruturados dos arquivos externos e trazê-los para a memória do Python de forma organizada.

**Funções necessárias:**

| Função | Descrição |
|---|---|
| `extract_bib_files(folder_path)` | Varre a pasta `data/`, usa o `bibtexparser` para ler cada `.bib`, padroniza os campos (remove chaves dos títulos, formata DOIs em minúsculo) e retorna um `DataFrame` do Pandas. |

---

## 3. `modules/processor.py` — Módulo de Transformação / Transform

> **Objetivo:** Garantir a integridade lógica e matemática dos dados. É o "faxineiro" e o "pesquisador" do sistema.

**Funções necessárias:**

| Função | Descrição |
|---|---|
| `clean_and_deduplicate(df)` | Encontra artigos com o mesmo DOI, mescla as bases de origem na coluna `indexers` (ex: `"Scopus, Ieee"`) e remove a linha duplicada. |
| `fetch_citations(df)` | Faz requisições HTTP via `requests` para a API do **Semantic Scholar** usando o DOI de cada artigo e retorna o número de citações atualizado. |

---

## 4. `modules/ai_gemini.py` — Módulo de Inteligência Artificial

> **Objetivo:** Isolar toda a comunicação com a API do Google Gemini. Se a lógica de IA mudar, apenas este arquivo precisa ser alterado.

**Funções necessárias:**

| Função | Descrição |
|---|---|
| `generate_query_and_justification(theme)` | Recebe o tema, monta um prompt rigoroso e retorna um JSON com a String de Busca (usando `AND`, `OR`, `NOT`) e a justificativa técnica. |
| `analyze_abstracts(df)` | Percorre o DataFrame, lê a coluna `abstract` de cada artigo, envia para a IA e retorna a análise crítica estruturada (Contribuição, Metodologia, Limitações). |

---

## 5. `modules/exporter.py` — Módulo de Saída / Load

> **Objetivo:** Pegar o DataFrame enriquecido e materializar nos formatos físicos exigidos para a entrega.
>
> ⚠️ **Status: a ser construído.**

**Funções necessárias:**

| Função | Descrição |
|---|---|
| `export_to_excel(df)` | Renomeia colunas do inglês para o português (ex: `title` → `Título`), formata e salva como `planilha.xlsx`. |
| `generate_comparative_analysis(df)` | Executa cálculos estatísticos com Pandas (ex: artigos por base, conferências vs. journals) e retorna um texto analítico pronto. |
| `generate_pdf_report(theme, query, analysis)` | Usa a biblioteca `fpdf2` para gerar o documento PDF final com os textos produzidos. |
| `create_final_zip()` | Compacta a planilha, o PDF e os arquivos `.bib` originais em um único arquivo `Entrega_Final.zip`. |