"""
Módulo de Inteligência Artificial (Google Gemini).

Responsabilidade: isolar toda a comunicação com a API do Google Gemini.
Qualquer mudança de modelo, prompt ou lógica de IA deve ser feita aqui.
"""

import json
import logging
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai

# Módulos não devem chamar basicConfig — a configuração é responsabilidade
# do ponto de entrada (main.py). Aqui apenas obtemos o logger do módulo.
logger = logging.getLogger(__name__)

# Cadeia de modelos tentadas em ordem. Quando um modelo retorna 429 (cota
# esgotada) o sistema avança automaticamente para o próximo.
# Referência: https://ai.google.dev/gemini-api/docs/models
_MODELS: list[str] = [
    "gemini-2.5-flash",       # melhor custo-benefício atual (estável)
    "gemini-2.5-flash-lite",  # mais leve da família 2.5 (estável)
]


# ---------------------------------------------------------------------------
# Inicialização lazy do cliente (evita erro de import quando a chave não está
# configurada durante testes que não usam este módulo)
# ---------------------------------------------------------------------------

def _get_client() -> genai.Client:
    """Retorna o cliente Gemini, inicializando-o na primeira chamada."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY não encontrada. Defina a variável no arquivo .env."
        )
    return genai.Client(api_key=api_key)


def _is_quota_error(exc: Exception) -> bool:
    """Retorna True se a exceção indicar limite de cota (HTTP 429)."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def _generate_with_fallback(client: genai.Client, prompt: str) -> str:
    """Tenta gerar conteúdo iterando pela cadeia de modelos em ``_MODELS``.

    Se um modelo retornar erro de cota (429 / RESOURCE_EXHAUSTED), o próximo
    da lista é utilizado automaticamente. Lança exceção apenas quando todos
    os modelos falharem.

    Returns:
        Texto bruto retornado pelo modelo que teve sucesso.
    """
    last_exc: Exception = RuntimeError("Nenhum modelo disponível em _MODELS.")
    for model in _MODELS:
        try:
            logger.debug("Tentando modelo '%s'...", model)
            response = client.models.generate_content(model=model, contents=prompt)
            logger.debug("Modelo '%s' respondeu com sucesso.", model)
            return response.text
        except Exception as exc:  # noqa: BLE001
            if _is_quota_error(exc):
                logger.warning(
                    "Modelo '%s' atingiu o limite de cota (429). Tentando próximo...",
                    model,
                )
                last_exc = exc
                time.sleep(1)  # pequena pausa antes de tentar o próximo
                continue
            raise  # erros que não são de cota são relançados imediatamente
    raise last_exc


def _parse_json_response(text: str) -> dict:
    """Remove blocos de markdown e faz o parsedo JSON retornado pela IA.

    O Gemini às vezes envolve a resposta em ```json ... ``` mesmo quando
    instruído a não fazê-lo. Este método remove esses marcadores de forma
    robusta antes de deserializar.
    """
    clean = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    return json.loads(clean)


# ---------------------------------------------------------------------------
# Prompts por base de dados
# ---------------------------------------------------------------------------

def _build_prompt(theme: str, database: str) -> str:
    """Retorna o prompt adequado para a base de dados solicitada."""

    base = database.strip().upper()

    if base == "IEEE":
        return f"""
            Você é um especialista em bibliometria e ciência da informação, focado em criar strings de busca avançadas de alta precisão.

            Sua tarefa é criar uma query de busca OTIMIZADA para a base de dados IEEE Xplore (Command Search) sobre o seguinte tema de pesquisa: "{theme}".

            REGRAS ESTRITAS PARA A QUERY NO IEEE XPLORE:
            1. Delimitadores de Campo: Use a sintaxe específica do IEEE especificando os campos de metadados. Englobe os blocos de busca usando "Document Title": e "Abstract":. Exemplo: ("Document Title": termo OR "Abstract": termo).
            2. Agrupamento: Agrupe sinônimos usando a cláusula OR dentro de parênteses. Use aspas duplas ("") estritamente para termos compostos exatos. O limite máximo de aninhamento são 3 níveis de parênteses.
            3. Truncamento Limitado (Regra Crítica): O motor do IEEE falha com excesso de wildcards. Use o asterisco (*) APENAS para plurais indispensáveis. Limite rigorosamente a no máximo 4 asteriscos (*) em toda a query.
            4. Controle de Siglas: Se o tema gerar siglas de 3 ou 4 letras, force um contexto para evitar falsos positivos. Exemplo: ("Sigla" AND "palavra de contexto").
            5. Exclusão Simples: Crie um bloco AND NOT no final para excluir termos indesejados, mas mantenha-o linear e sem parênteses complexos internos.

            DIRETRIZES DE SAÍDA (FORMATO):
            Retorne ÚNICA e EXCLUSIVAMENTE um objeto JSON válido.
            NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown (como ```json).
            As aspas duplas dentro dos valores do JSON devem ser corretamente escapadas (\\"), especialmente as aspas que fazem parte da sintaxe de busca do IEEE.

            Estrutura EXATA exigida:
            {{
            "query": "string de busca final formatada em uma única linha",
            "justification": "Breve explicação (em tópicos) das palavras principais, e como a limitação de wildcards e aninhamento do IEEE foi tratada."
            }}
        """

    if base == "SCOPUS":
        return f"""
            Você é um especialista em bibliometria e ciência da informação, focado em criar strings de busca avançadas de alta precisão.

            Sua tarefa é criar uma query de busca OTIMIZADA para o Scopus sobre o seguinte tema de pesquisa: "{theme}".

            REGRAS ESTRITAS PARA A QUERY:
            1. Delimitadores: Use obrigatoriamente a sintaxe TITLE-ABS-KEY( ) para restringir a busca a Títulos, Resumos e Palavras-chave.
            2. Agrupamento: Agrupe sinônimos usando a cláusula OR dentro de parênteses. Use aspas duplas ("") estritamente para termos compostos exatos.
            3. Truncamento: Use o asterisco (*) para capturar plural e variações de sufixos (ex: algorithm*).
            4. Controle de Siglas (Crucial): Se o tema gerar siglas de 3 ou 4 letras, force um contexto para evitar falsos positivos. Exemplo: ( "Sigla" AND palavra_de_contexto* ).
            5. Exclusão: Crie um bloco genérico AND NOT no final para excluir áreas do conhecimento ou termos homônimos não relacionados ao escopo provável do tema.

            DIRETRIZES DE SAÍDA (FORMATO):
            Retorne ÚNICA e EXCLUSIVAMENTE um objeto JSON válido.
            NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown (como ```json).
            As aspas duplas dentro dos valores do JSON devem ser corretamente escapadas (\\").

            Estrutura EXATA exigida:
            {{
            "query": "string de busca final formatada em uma única linha",
            "justification": "Breve explicação (em tópicos) das palavras principais, sinônimos, controle de contexto e exclusões aplicadas."
            }}
        """

    if base == "ACM":
        return f"""
            Você é um especialista em bibliometria e ciência da informação, focado em criar strings de busca avançadas de alta precisão.

            Sua tarefa é criar uma query de busca OTIMIZADA para a base de dados ACM Digital Library sobre o seguinte tema de pesquisa: "{theme}".

            REGRAS ESTRITAS PARA A QUERY NA ACM:
            1. Delimitadores de Campo: A ACM exige a declaração explícita dos campos. Para buscar de forma abrangente, aplique os blocos de conceitos aos campos Title, Abstract e Keyword. A estrutura deve ser: (Title:("termo1" OR termo2) OR Abstract:("termo1" OR termo2) OR Keyword:("termo1" OR termo2)).
            2. Operadores Booleanos: Todos os operadores (AND, OR, NOT) devem estar OBRIGATORIAMENTE em LETRAS MAIÚSCULAS. A ACM falha ou ignora booleanos em minúsculas.
            3. Agrupamento e Aspas: Use aspas duplas ("") estritamente para termos compostos (ex: "software architecture"). Agrupe a lógica sempre com parênteses.
            4. Truncamento (Regra Crítica): Use o asterisco (*) para variações de palavras, mas NUNCA coloque um asterisco dentro de aspas duplas (ex: "quantum comput*" causará erro na ACM). O asterisco só funciona em palavras simples soltas.
            5. Controle de Siglas e Exclusão: Pareie siglas curtas com palavras de contexto usando AND. Para exclusões, use NOT no final da query para remover termos fora do escopo.

            DIRETRIZES DE SAÍDA (FORMATO):
            Retorne ÚNICA e EXCLUSIVAMENTE um objeto JSON válido.
            NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown (como ```json).
            As aspas duplas dentro dos valores do JSON devem ser corretamente escapadas (\\"), para não quebrar o parser da aplicação.

            Estrutura EXATA exigida:
            {{
            "query": "string de busca final formatada em uma única linha",
            "justification": "Breve explicação (em tópicos) das palavras principais e como a sintaxe específica da ACM (Title/Abstract/Keyword) foi aplicada."
            }}
        """

    raise ValueError(
        f"Base de dados '{database}' não suportada. Use: 'Scopus', 'IEEE' ou 'ACM'."
    )


# ---------------------------------------------------------------------------
# Funções públicas do módulo
# ---------------------------------------------------------------------------

def generate_query_and_justification(theme: str, database: str) -> tuple[str, str] | tuple[None, None]:
    """Gera uma string de busca otimizada e sua justificativa técnica para
    a base de dados solicitada.

    Args:
        theme: Tema da pesquisa (ex: ``"Post-Quantum Cryptography"``).
        database: Base de dados alvo. Valores aceitos: ``"Scopus"``, ``"IEEE"``, ``"ACM"``.

    Returns:
        Tupla ``(query, justification)`` com as strings geradas pela IA, ou
        ``(None, None)`` em caso de falha.
    """
    logger.info("Gerando query para '%s' (base: %s)...", theme, database)

    prompt = _build_prompt(theme, database)

    try:
        client = _get_client()
        text    = _generate_with_fallback(client, prompt)
        ai_data = _parse_json_response(text)
        logger.info("Query gerada com sucesso.")
        return ai_data["query"], ai_data["justification"]

    except (ValueError, KeyError) as exc:
        logger.error("Erro ao processar resposta da IA: %s", exc)
    except Exception as exc:
        logger.error("Erro ao comunicar com o Gemini: %s", exc)

    return None, None


def analyze_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    """Envia os abstracts para a IA gerar a análise crítica estruturada.

    Para cada artigo com abstract válido, consulta o Gemini e armazena
    o resultado formatado em 5 tópicos na coluna ``observations``.

    Args:
        df: DataFrame (preferencialmente já deduplicado e enriquecido).

    Returns:
        DataFrame com a coluna ``observations`` adicionada.
    """
    if df.empty:
        logger.warning("DataFrame vazio. Pulando análise de abstracts.")
        return df

    df = df.copy()
    df["observations"] = None

    client = _get_client()
    total = df["abstract"].notna().sum()
    logger.info("Iniciando leitura crítica com IA para %d artigos...", total)

    for idx, row in df.iterrows():
        abstract = row.get("abstract", "")
        title    = row.get("title", "Artigo sem título")

        if not abstract or len(str(abstract)) < 50:
            df.at[idx, "observations"] = "Resumo ausente ou muito curto para análise."
            continue

        logger.info("Analisando: '%s'...", str(title)[:60])

        prompt = f"""
        Leia o abstract deste artigo científico.
        Eu preciso que você extraia uma análise crítica estruturada.

        Retorne APENAS um JSON estrito com uma única chave chamada "observations".
        O valor dessa chave deve ser um texto contendo exatamente estes 5 tópicos (use quebra de linha \\n):
        - Contribuição principal: ...
        - Metodologia usada: ...
        - Limitações identificadas: ...
        - Potencial de aplicação: ...
        - Relevância para pesquisa futura: ...

        NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown.

        Abstract: {abstract}
        """

        try:
            text    = _generate_with_fallback(client, prompt)
            ai_data = _parse_json_response(text)
            df.at[idx, "observations"] = ai_data.get("observations", "Erro na formatação da resposta.")
        except Exception as exc:
            logger.warning("Erro ao analisar abstract (índice %d): %s", idx, exc)
            df.at[idx, "observations"] = "Erro na análise da IA."

        # Delay obrigatório para não estourar a cota gratuita da API
        time.sleep(4)

    logger.info("Leitura crítica finalizada.")
    return df