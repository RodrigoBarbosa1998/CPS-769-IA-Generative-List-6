# Análise da Qualidade de Experiência (QoE) em Streaming de Vídeo com Function Calling e Chain of Thought

## Descrição

Este projeto apresenta uma aplicação desenvolvida para analisar a Qualidade de Experiência (QoE) em serviços de streaming de vídeo. A aplicação utiliza técnicas avançadas de processamento de linguagem natural, especificamente \textit{Function Calling} e \textit{Chain of Thought} (CoT), por meio das APIs da OpenAI, para fornecer respostas precisas e contextualizadas em linguagem natural a partir dos dados fornecidos. A abordagem permite responder a diversas perguntas relacionadas ao desempenho de vídeo, como a pior qualidade de recepção, estratégias de troca de servidor e o impacto da latência na experiência do usuário.

## Objetivo

O objetivo deste trabalho foi desenvolver uma aplicação que:
1. Aceite perguntas dos usuários em linguagem natural relacionadas à Qualidade de Experiência (QoE) na transmissão de vídeo.
2. Forneça respostas detalhadas com base em análises e cálculos em tempo real, utilizando \textit{Function Calling} e \textit{Chain of Thought}.

## Tecnologias Utilizadas

- **Linguagem de Programação**: Python
- **Interface Web**: Dash
- **Processamento de Linguagem Natural**: OpenAI API (modelo `gpt-4o-mini`)
- **Manipulação de Dados**: Pandas, NumPy
- **Machine Learning**: Métricas de similaridade com `sklearn`

## Estrutura do Projeto

O projeto contém os seguintes componentes principais:

1. **`main.py`**: Código principal da aplicação, que inclui a configuração da interface, processamento de dados, e implementação do \textit{Function Calling} e do \textit{Chain of Thought}.

2. **Dados**: Dois arquivos CSV são fornecidos para análise:
    - `bitrate_train.csv`: Contém medições da taxa de transmissão (\textit{bitrate}) de vídeo.
    - `rtt_train.csv`: Contém medições de latência (\textit{RTT}) em milissegundos.

3. **Interface de Usuário**: Desenvolvida em Dash, permite que os usuários insiram perguntas em linguagem natural e recebam respostas baseadas em análises dos dados.

4. **Funções Implementadas**: Diversas funções foram implementadas para responder a diferentes tipos de perguntas, como:
    - Qual cliente tem a pior qualidade de recepção de vídeo?
    - Qual é o tempo médio de latência de um cliente específico?
    - Como o bitrate médio varia ao longo do dia para um servidor específico?

## Como Funciona

1. O usuário faz uma pergunta na interface web.
2. A pergunta é processada para identificar o tipo de análise ou cálculo necessário.
3. A aplicação utiliza a API da OpenAI para analisar a pergunta e chamar a função adequada (via \textit{Function Calling}).
4. A função realiza a análise dos dados fornecidos, como cálculos de QoE, variações de latência e taxa de transmissão.
5. O resultado é exibido ao usuário junto com uma explicação detalhada, utilizando a técnica de \textit{Chain of Thought} para fornecer uma resposta clara e lógica.

## Como Executar

1. Clone este repositório:
    ```bash
    git clone https://github.com/seu-usuario/nome-do-repositorio.git
    ```
2. Navegue até o diretório do projeto:
    ```bash
    cd nome-do-repositorio
    ```
3. Instale as dependências:

4. Execute o arquivo principal:
    ```bash
    python main.py
    ```
5. Abra o navegador e acesse a aplicação no endereço `http://127.0.0.1:8050`.

## Requisitos

- Python 3.8+
- Chave da API da OpenAI
- Pacotes listados

## Resultados

Os resultados das análises são apresentados em linguagem natural, com uma explicação detalhada sobre como foram obtidos. A aplicação é capaz de fornecer insights sobre a qualidade do serviço de streaming, como identificar o cliente com a pior QoE, calcular a variação do bitrate ao longo do dia e avaliar o impacto do aumento da latência.

## Lições Aprendidas

- **Uso do Function Calling**: Permite a integração eficiente de operações externas, otimizando a análise dos dados sem a necessidade de re-treinar o modelo de linguagem.
- **Aplicação do Chain of Thought (CoT)**: A implementação do CoT foi fundamental para estruturar o raciocínio e dividir problemas complexos em etapas menores, resultando em respostas mais precisas e explicativas.