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
from pathlib import Path

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
    # Carrega .env do diretório raiz do projeto, não do cwd atual
    # override=True garante que o .env sobrescreve variáveis já definidas no ambiente
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
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
            Você é um especialista em bibliometria e ciência da informação, focado em criar strings de busca avançadas, CONCISAS e de altíssima precisão.

            Sua tarefa é extrair os conceitos centrais do tema "{theme}" e criar uma query de busca OTIMIZADA para a base IEEE Xplore (Command Search).

            REGRAS ESTRITAS PARA A QUERY NO IEEE XPLORE:
            1. Máxima Relevância (TÍTULO E RESUMO): Para evitar artigos genéricos ou de baixa relevância, o conceito principal DEVE ser buscado obrigatoriamente no título do documento usando a tag "Document Title". Os conceitos secundários e de contexto devem ser buscados no resumo usando a tag "Abstract".
            2. Sintaxe de Campos Obrigatória (ALERTA CRÍTICO): O motor do IEEE NÃO interpreta múltiplos termos agrupados em uma única declaração de campo. Você DEVE repetir a tag para CADA termo individualmente.
               -> PROIBIDO: "Abstract":("termo A" OR "termo B")
               -> OBRIGATÓRIO: ("Abstract":"termo A" OR "Abstract":"termo B")
            3. Controle de Tamanho e Operadores: Extraia 1 conceito principal e no máximo 2 conceitos secundários (com até 1 sinônimo cada). A query DEVE conter os operadores AND, OR e NOT.
            4. Limitação de Curingas: Use no MÁXIMO 2 asteriscos (*) em toda a string. O motor do IEEE falha com excesso de wildcards. Prefira escrever variações essenciais por extenso se necessário.
            5. Estrutura Exigida: A sua query DEVE seguir exatamente este esqueleto lógico, respeitando o limite de parênteses:
               ("Document Title":"conceito principal exato") AND (""Document Title":"conceito principal exato" OR "Abstract":"conceito secundário" OR "Abstract":"sinônimo do secundário") AND NOT ("Document Title":"termo irrelevante" OR "Abstract":"termo irrelevante")

            DIRETRIZES DE SAÍDA (FORMATO JSON):
            Retorne ÚNICA e EXCLUSIVAMENTE um objeto JSON válido.
            NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown.
            As aspas duplas dentro dos valores do JSON devem ser corretamente escapadas (\\").
            
            REGRA CRÍTICA PARA A JUSTIFICATIVA: 
            A justificativa deve ser uma única string contendo exatamente 4 tópicos. Use explicitamente '\\n' para criar as quebras de linha dentro do JSON. Não use quebras de linha reais.

            Estrutura EXATA exigida:
            {{
            "query": "string de busca final formatada em uma única linha, seguindo o esqueleto lógico e a repetição de tags",
            "justification": "- Palavras-chave: [Conceito principal focado no Document Title]\\n- Sinônimos e Contexto: [Quais sinônimos usou no Abstract e como agrupou]\\n- Sintaxe da Base: [Uso repetido das tags de campo e limite de wildcards]\\n- Exclusões: [Quais termos removeu com NOT]"
            }}
        """

    if base == "SCOPUS":
        return f"""
            Você é um especialista em bibliometria e ciência da informação, focado em criar strings de busca avançadas, CONCISAS e de altíssima precisão.

            Sua tarefa é extrair os conceitos centrais do tema "{theme}" e criar uma query de busca OTIMIZADA para o Scopus.

            REGRAS ESTRITAS PARA A QUERY NO SCOPUS:
            1. Precisão com ABS() (OBRIGATÓRIO): Para garantir que a busca seja estritamente sobre o tema e não genérica, o conceito principal DEVE ser buscado especificamente no resumo usando ABS(). 
            2. Operadores Obrigatórios: A sua query final DEVE obrigatoriamente conter pelo menos um operador AND, um OR e um AND NOT.
            3. Limite de Sinônimos (CONTROLE DE TAMANHO): Extraia no máximo 2 conceitos secundários do tema. Para cada conceito secundário, forneça NO MÍNIMO 1 e NO MÁXIMO 2 sinônimos diretos usando o operador OR. Não invente termos periféricos.
            4. Regra das Aspas (ALERTA CRÍTICO): QUALQUER termo composto por duas ou mais palavras DEVE OBRIGATORIAMENTE estar entre aspas duplas, mesmo com asterisco (ex: "machine learning*").
            5. Estrutura Exigida: A sua query DEVE seguir exatamente este esqueleto lógico:
               ABS("conceito principal exato") AND TITLE-ABS-KEY("conceito principal exato" OR "conceito secundário" OR "sinônimo do secundário") AND NOT TITLE-ABS-KEY("termo irrelevante 1" OR "termo irrelevante 2")

            DIRETRIZES DE SAÍDA (FORMATO JSON):
            Retorne ÚNICA e EXCLUSIVAMENTE um objeto JSON válido.
            NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown.
            As aspas duplas dentro dos valores do JSON devem ser corretamente escapadas (\\").
            
            A justificativa deve ser uma única string contendo exatamente 4 tópicos separados explicitamente por '\\n'.

            Estrutura EXATA exigida:
            {{
            "query": "string de busca final formatada em uma única linha seguindo o esqueleto lógico",
            "justification": "- Palavras-chave: [Conceito principal no ABS]\\n- Sinônimos e Contexto: [Quais sinônimos usou no OR]\\n- Sintaxe da Base: [Uso do ABS, AND, OR e aspas]\\n- Exclusões: [Quais termos removeu com AND NOT para evitar viés]"
            }}
        """

    if base == "ACM":
        return f"""
            Você é um especialista em bibliometria e ciência da informação, focado em criar strings de busca avançadas, CONCISAS e de altíssima precisão.

            Sua tarefa é extrair os conceitos centrais do tema "{theme}" e criar uma query de busca OTIMIZADA para a base ACM Digital Library.

            REGRAS ESTRITAS PARA A QUERY NA ACM:
            1. Máxima Relevância (TÍTULO E RESUMO): Para evitar excesso de artigos irrelevantes, o conceito principal DEVE ser buscado OBRIGATORIAMENTE no campo Title. Os conceitos secundários devem ser buscados no campo Abstract. 
               -> PROIBIDO buscar tudo em todos os campos simultaneamente.
            2. Truncamento e Aspas (ALERTA CRÍTICO DE FALHA): A ACM gera erro fatal se houver asterisco (*) dentro de aspas duplas. 
               -> PROIBIDO: "software architecture*"
               -> PERMITIDO: "software architecture" OR architect*
            3. Limite de Conceitos: Extraia 1 conceito principal (para o título) e no máximo 2 conceitos secundários (com até 1 sinônimo cada, para o resumo).
            4. Operadores Booleanos: Todos os operadores DEVEM estar em MAIÚSCULAS (AND, OR, NOT). Na ACM, use apenas NOT (e não AND NOT) para exclusões.
            5. Estrutura Exigida: A sua query DEVE seguir exatamente este esqueleto lógico, respeitando a sintaxe da base:
               Title:("conceito principal exato") AND Abstract:("conceito principal exato" OR "conceito secundário" OR "sinônimo do secundário") NOT Title:("termo irrelevante 1" OR "termo irrelevante 2")

            DIRETRIZES DE SAÍDA (FORMATO JSON):
            Retorne ÚNICA e EXCLUSIVAMENTE um objeto JSON válido.
            NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown.
            As aspas duplas dentro dos valores do JSON devem ser corretamente escapadas (\\").
            
            REGRA CRÍTICA PARA A JUSTIFICATIVA: 
            A justificativa deve ser uma única string contendo exatamente 4 tópicos. Use explicitamente '\\n' para criar as quebras de linha dentro do JSON. Não use quebras de linha reais.

            Estrutura EXATA exigida:
            {{
            "query": "string de busca final formatada em uma única linha, seguindo rigorosamente o esqueleto lógico da Regra 5",
            "justification": "- Palavras-chave: [Conceito principal focado no Title]\\n- Sinônimos e Contexto: [Como agrupou no Abstract e lidou com a regra do curinga]\\n- Sintaxe da Base: [Declaração explícita de campos e uso correto das aspas sem asterisco]\\n- Exclusões: [Quais termos removeu com NOT]"
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


def analyze_abstracts(df: pd.DataFrame, on_step=None) -> pd.DataFrame:
    """Envia os abstracts para a IA gerar a análise crítica estruturada.

    Para cada artigo com abstract válido, consulta o Gemini e armazena
    o resultado formatado em 5 tópicos na coluna ``observations``.

    Args:
        df:      DataFrame (preferencialmente já deduplicado e enriquecido).
        on_step: Callback opcional ``(mensagem, percentual)`` para progresso na UI.
                 Percentual -1 = aviso, -2 = info de espera por cota.

    Returns:
        DataFrame com a coluna ``observations`` adicionada.
    """
    if df.empty:
        logger.warning("DataFrame vazio. Pulando análise de abstracts.")
        return df

    df = df.copy()
    df["observations"] = None

    client = _get_client()
    total = int(df["abstract"].notna().sum())
    logger.info("Iniciando leitura crítica com IA para %d artigos...", total)

    done = 0
    for idx, row in df.iterrows():
        abstract = row.get("abstract", "")
        title    = row.get("title", "Artigo sem título")

        if not abstract or len(str(abstract)) < 50:
            df.at[idx, "observations"] = "Resumo ausente ou muito curto para análise."
            continue

        done += 1
        pct_start = int((done - 1) / max(total, 1) * 100)
        short_title = str(title)[:60]

        logger.info("Analisando: '%s'...", short_title)
        if on_step:
            on_step(f"🔍 ({done}/{total}) Analisando: '{short_title}'...", pct_start)

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
            pct_done = int(done / max(total, 1) * 100)
            if on_step:
                on_step(f"✅ ({done}/{total}) Concluído: '{short_title}'", pct_done)
        except Exception as exc:
            logger.warning("Erro ao analisar abstract (índice %d): %s", idx, exc)
            df.at[idx, "observations"] = "Erro na análise da IA."
            if on_step:
                on_step(
                    f"⚠️ ({done}/{total}) Erro ao analisar '{str(title)[:40]}': {type(exc).__name__}",
                    -1,
                )

        # Delay obrigatório para não estourar a cota gratuita da API
        if done < total:
            if on_step:
                on_step(f"⏳ Aguardando 4s antes da próxima requisição...", -2)
            time.sleep(4)

    logger.info("Leitura crítica finalizada.")
    return df