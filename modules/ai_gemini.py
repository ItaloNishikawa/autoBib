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
            Você é um especialista em bibliometria e ciência da informação, focado em criar strings de busca avançadas de alta precisão.

            Sua tarefa é criar uma query de busca OTIMIZADA para a base de dados IEEE Xplore (Command Search) sobre o seguinte tema de pesquisa: "{theme}".

            REGRAS ESTRITAS PARA A QUERY NO IEEE XPLORE:
            1. SINTAXE DE CAMPOS (ALERTA CRÍTICO DE FALHA): O motor do IEEE NÃO interpreta corretamente múltiplos termos agrupados dentro de uma única declaração de campo. 
               -> PROIBIDO: "Document Title":("termo A" OR "termo B")
               -> OBRIGATÓRIO: Repita a tag de campo para CADA termo individualmente. Crie blocos combinando Título e Resumo para cada sinônimo. Exemplo exato: (("Document Title":"termo A" OR "Abstract":"termo A") OR ("Document Title":"termo B" OR "Abstract":"termo B")).
            2. LIMITE MÁXIMO DE TERMOS: O motor do IEEE aceita NO MÁXIMO 40 termos lógicos. Como a Regra 1 exige a repetição de campos, você DEVE ser extremamente seletivo. Escolha no máximo 4 a 5 sinônimos principais por conceito.
            3. LIMITAÇÃO DE CURINGAS: O motor sofre crash com excesso de wildcards. É ESTRITAMENTE PROIBIDO usar mais de 4 asteriscos (*) em toda a string. Prefira escrever variações gramaticais por extenso.
            4. Aninhamento Simples: NUNCA abra mais de 3 níveis de parênteses consecutivos (Ex: "((("). Mantenha a lógica de agrupamento o mais plana possível para evitar erro de timeout na base.
            5. Exclusão Simples: Crie um bloco AND NOT no final para excluir no máximo 3 termos indesejados. Exemplo: AND NOT ("termo ruim" OR "outro termo").
            6. Operadores e Campos Obrigatórios: Todos os operadores booleanos/de proximidade DEVEM estar em MAIÚSCULAS (AND, OR, NOT, NEAR, ONEAR). Há um máximo de 25 termos de busca por cláusula de busca. O nome do campo de dados DEVE ser declarado antes de cada termo de busca individual, conforme exigido pela sintaxe do IEEE Command Search.

            DIRETRIZES DE SAÍDA (FORMATO):
            Retorne ÚNICA e EXCLUSIVAMENTE um objeto JSON válido.
            NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown.
            As aspas duplas dentro dos valores do JSON devem ser corretamente escapadas (\\").
            
            REGRA CRÍTICA PARA A JUSTIFICATIVA: 
            A justificativa deve ser uma única string contendo exatamente 4 tópicos. Use explicitamente '\\n' para criar as quebras de linha dentro do JSON. Não use quebras de linha reais.

            Estrutura EXATA exigida:
            {{
            "query": "string de busca final formatada em uma única linha, seguindo a Regra 1 rigorosamente",
            "justification": "- Palavras-chave: [Liste as principais]\\n- Sinônimos e Contexto: [Como agrupou e evitou falsos positivos]\\n- Sintaxe da Base: [Como aplicou as regras específicas desta base (ex: wildcards, campos)]\\n- Exclusões: [Quais termos removeu com NOT]"
            }}
        """

    if base == "SCOPUS":
        return f"""
            Você é um especialista em bibliometria e ciência da informação, focado em criar strings de busca avançadas de alta precisão.

            Sua tarefa é criar uma query de busca OTIMIZADA para o Scopus sobre o seguinte tema de pesquisa: "{theme}".

            REGRAS ESTRITAS PARA A QUERY NO SCOPUS:
            1. Delimitadores: Use obrigatoriamente a sintaxe TITLE-ABS-KEY() no início do bloco de inclusão e TITLE-ABS-KEY() no bloco de exclusão.
            2. Regra das Aspas (ALERTA CRÍTICO): No Scopus, o espaço atua como operador AND. Portanto, QUALQUER termo composto por duas ou mais palavras DEVE OBRIGATORIAMENTE estar entre aspas duplas, MESMO que contenha um asterisco no final. 
               -> ERRADO: key exchange*
               -> CORRETO: "key exchange*"
            3. Balanceamento de Parênteses (ALERTA CRÍTICO): Evite erros de sintaxe mantendo a estrutura plana. NUNCA aninhe mais de 2 níveis de parênteses. A string inteira deve ser essencialmente: TITLE-ABS-KEY((grupo 1) OR (grupo 2)) AND NOT TITLE-ABS-KEY((exclusões)). Certifique-se matematicamente de que cada parêntese aberto seja fechado.
            4. Truncamento: Use o asterisco (*) para capturar plural e variações de sufixos (ex: "algorithm*").
            5. Controle de Siglas: Se o tema gerar siglas de 3 ou 4 letras, force um contexto para evitar falsos positivos. Exemplo: ("Sigla" AND "palavra de contexto*").
            6. Exclusão: Crie um bloco genérico AND NOT TITLE-ABS-KEY(...) no final para excluir áreas não relacionadas.

            DIRETRIZES DE SAÍDA (FORMATO):
            Retorne ÚNICA e EXCLUSIVAMENTE um objeto JSON válido.
            NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown.
            As aspas duplas dentro dos valores do JSON devem ser corretamente escapadas (\\").
            
            REGRA CRÍTICA PARA A JUSTIFICATIVA: 
            A justificativa deve ser uma única string contendo exatamente 4 tópicos. Use explicitamente '\\n' para criar as quebras de linha dentro do JSON. Não use quebras de linha reais.

            Estrutura EXATA exigida:
            {{
            "query": "string de busca final formatada em uma única linha",
            "justification": "- Palavras-chave: [Liste as principais]\\n- Sinônimos e Contexto: [Como agrupou e evitou falsos positivos]\\n- Sintaxe da Base: [Como aplicou as regras específicas desta base (ex: wildcards, campos)]\\n- Exclusões: [Quais termos removeu com NOT]"
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
            As aspas duplas dentro dos valores do JSON devem ser corretamente escapadas (\\").
            
            REGRA CRÍTICA PARA A JUSTIFICATIVA: 
            A justificativa deve ser uma única string contendo exatamente 4 tópicos. 
            Use explicitamente os caracteres '\\n' (barra invertida e a letra n) para criar as quebras de linha dentro do JSON. Não use quebras de linha reais.

            Estrutura EXATA exigida:
            {{
            "query": "string de busca final formatada em uma única linha",
            "justification": "- Palavras-chave: [Liste as principais]\\n- Sinônimos e Contexto: [Como agrupou e evitou falsos positivos]\\n- Sintaxe da Base: [Como aplicou as regras específicas desta base (ex: wildcards, campos)]\\n- Exclusões: [Quais termos removeu com NOT]"
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