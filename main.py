import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Main:
    def __init__(self, api_key):
        # Configurar a chave da API da OpenAI
        self.client = OpenAI(api_key=api_key)

        # Carregar os dados das métricas
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.bitrate_data = pd.read_csv(os.path.join(self.data_dir, 'bitrate_train.csv'))
        self.latency_data = pd.read_csv(os.path.join(self.data_dir, 'rtt_train.csv'))

        # Realizar a pré-análise dos dados
        self.preprocessed_data = self.preprocess_data()

        # Frases de referência para identificação de perguntas
        self.reference_phrases = [
            # Perguntas relacionadas à pior qualidade de recepção de vídeo
            "pior qualidade de recepção de vídeo",
            "pior vídeo",
            "baixa qualidade na recepção do vídeo",
            "baixa qualidade na recepção da informação",
            "qualidade ruim no vídeo",
            "vídeo ruim",
            "cliente com vídeo ruim",
            "qual cliente tem a pior qualidade de recepção de vídeo ao longo do tempo",

            # Perguntas relacionadas à QoE mais consistente
            "qual servidor fornece a QoE mais consistente",
            "servidor com a menor variação de QoE",
            "servidor mais estável em QoE",

            # Perguntas relacionadas à melhor estratégia de troca de servidor
            "melhor estratégia de troca de servidor para maximizar a qualidade de experiência do cliente",
            "qual é a melhor estratégia de troca para o cliente",
            "como melhorar a experiência do cliente mudando de servidor",

            # Perguntas relacionadas ao aumento da latência
            "se a latência aumentar",
            "como o aumento da latência afeta a QoE",
            "efeito do aumento de latência na QoE do cliente",

            # Novas perguntas: tempo médio de latência do cliente
            "qual é o tempo médio de latência do cliente",
            "tempo médio de latência para o cliente",
            "latência média do cliente",
            
            # Novas perguntas: bitrate médio para o servidor
            "qual é o bitrate médio para o servidor",
            "bitrate médio para o servidor",
            "média de bitrate do servidor",

            # Novas perguntas: maior variação de bitrate do cliente
            "qual foi a maior variação de bitrate para o cliente",
            "maior variação de bitrate para o cliente",
            "variação máxima de bitrate do cliente",

            # Novas perguntas: menor latência registrada para o cliente
            "qual foi a menor latência registrada para o cliente",
            "menor latência do cliente",
            "latência mínima registrada para o cliente",

            # Novas perguntas: variação de bitrate médio ao longo do dia para o servidor
            "como o bitrate médio varia ao longo do dia para o servidor",
            "variação do bitrate ao longo do dia para o servidor",
            "bitrate médio do servidor ao longo do dia"
        ]

        self.reference_embeddings = self.generate_embeddings(self.reference_phrases)

        # Inicializar a aplicação Dash
        self.app = dash.Dash(__name__)
        self.app.title = "CPS769"

        # Configurar o layout da aplicação
        self.app.layout = self.create_layout()

        # Configurar o callback
        self.setup_callbacks()

    def preprocess_data(self):
        # Converter timestamp para granularidade de minuto
        self.bitrate_data['timestamp'] = pd.to_datetime(self.bitrate_data['timestamp'], unit='s')
        self.bitrate_data['minute'] = self.bitrate_data['timestamp'].dt.floor('min')
        
        self.latency_data['timestamp'] = pd.to_datetime(self.latency_data['timestamp'], unit='s')
        self.latency_data['minute'] = self.latency_data['timestamp'].dt.floor('min')

        # Agrupar por cliente, servidor e minuto, calculando a média
        bitrate_grouped = self.bitrate_data.groupby(['client', 'server', 'minute'])['bitrate'].mean().reset_index()
        latency_grouped = self.latency_data.groupby(['client', 'server', 'minute'])['rtt'].mean().reset_index()

        # Mesclar as duas séries com base em cliente, servidor e minuto
        merged_data = pd.merge(bitrate_grouped, latency_grouped, on=['client', 'server', 'minute'])

        # Normalizar as colunas de bitrate e latência
        merged_data['Normalized_Bitrate'] = self.normalize_data(merged_data, 'bitrate')
        merged_data['Normalized_Latency'] = self.normalize_data(merged_data, 'rtt')

        # Calcular a QoE normalizada
        merged_data['QoE'] = merged_data['Normalized_Bitrate'] / merged_data['Normalized_Latency']

        return merged_data

    def normalize_data(self, data, column_name):
        # Normalizar os dados usando normalização min-max
        min_value = data[column_name].min()
        max_value = data[column_name].max()
        return (data[column_name] - min_value) / (max_value - min_value)

    def generate_embeddings(self, texts):
        # Gera embeddings para uma lista de textos
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return [embedding.embedding for embedding in response.data]

    def find_most_similar(self, question_embedding):
        # Calcula a similaridade entre a pergunta e as frases de referência
        similarities = cosine_similarity([question_embedding], self.reference_embeddings)
        most_similar_index = np.argmax(similarities)
        most_similar_score = similarities[0][most_similar_index]
        return self.reference_phrases[most_similar_index], most_similar_score

    def create_layout(self):
        # Cria o layout da aplicação Dash
        return html.Div(
            style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f4f4', 'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'},
            children=[
                html.Div(
                    style={'backgroundColor': '#1f77b4', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'},
                    children=[
                        html.H1("CPS769 - Introduction to Artificial Intelligence and Generative Learning", style={'color': 'white', 'textAlign': 'center'}),
                    ]
                ),
                html.Div(
                    style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.2)', 'marginBottom': '20px'},
                    children=[
                        html.Label('Digite sua pergunta:', style={'fontWeight': 'bold', 'fontSize': '16px'}),
                        dcc.Input(
                            id='input-question',
                            type='text',
                            placeholder='Ex.: Qual cliente tem a pior qualidade na aplicação de vídeo streaming?',
                            style={'width': '100%', 'padding': '10px', 'marginTop': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
                        ),
                        html.Button('Enviar', id='submit-button', n_clicks=0,
                                    style={'marginTop': '20px', 'padding': '10px 20px', 'borderRadius': '5px', 'backgroundColor': '#1f77b4', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
                        html.Div(id='status-message', style={'marginTop': '10px', 'fontSize': '14px', 'color': '#888'})
                    ]
                ),
                html.Div(
                    style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.2)'},
                    children=[
                        html.H2('Resposta:', style={'fontSize': '18px', 'marginBottom': '10px'}),
                        html.Div(id='output-response', style={'fontSize': '16px', 'color': '#333'})
                    ]
                )
            ]
        )

    def setup_callbacks(self):
        # Configura os callbacks para a aplicação Dash
        @self.app.callback(
            Output('output-response', 'children'),
            Output('status-message', 'children'),
            [Input('submit-button', 'n_clicks')],
            [State('input-question', 'value')]
        )
        def process_question(n_clicks, question):
            if n_clicks > 0 and question:
                status_message = "Processando a pergunta..."
                try:
                    # Obter o embedding da pergunta
                    question_embedding = self.generate_embeddings([question])[0]
                    most_similar_phrase, most_similar_score = self.find_most_similar(question_embedding)

                    # Definir um limiar de similaridade; por exemplo, 0.86 (ajustável)
                    similarity_threshold = 0.86

                    if most_similar_score >= similarity_threshold:
                        # Utilizar function calling com GPT
                        return self.process_question_with_function_calling(question)
                    else:
                        answer = "Desculpe, não tenho a resposta para essa pergunta. Minha funcionalidade está limitada a analisar a qualidade de experiência na recepção de vídeo."

                    status_message = "Pergunta processada com sucesso."
                    return answer, status_message
                except Exception as e:
                    status_message = "Erro ao processar a pergunta."
                    return f"Erro: {e}", status_message
            return "Por favor, insira uma pergunta e clique em 'Enviar'.", ""

    def process_question_with_function_calling(self, question):
        # Usar GPT para analisar a pergunta e decidir qual função chamar
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em análise de qualidade de experiência (QoE). Aqui estão as funções disponíveis: 'calculate_qoe_with_explanation', 'calculate_most_consistent_server', 'calculate_best_server_switch_strategy', 'calculate_qoe_with_increased_latency', 'calculate_avg_latency_for_client', 'calculate_avg_bitrate_for_server', 'calculate_max_bitrate_variation_for_client', 'calculate_min_latency_for_client', 'calculate_bitrate_variation_for_server'."},
                {"role": "user", "content": question}
            ],
            functions=[
                {"name": "calculate_qoe_with_explanation", "description": "Calcula o cliente com a pior QoE."},
                {"name": "calculate_most_consistent_server", "description": "Calcula o servidor com a QoE mais consistente."},
                {"name": "calculate_best_server_switch_strategy", "description": "Encontra a melhor estratégia de troca de servidor para um cliente específico."},
                {"name": "calculate_qoe_with_increased_latency", "description": "Calcula o impacto do aumento de latência na QoE de um cliente específico."},
                {"name": "calculate_avg_latency_for_client", "description": "Calcula o tempo médio de latência de um cliente específico."},
                {"name": "calculate_avg_bitrate_for_server", "description": "Calcula o bitrate médio para um servidor específico."},
                {"name": "calculate_max_bitrate_variation_for_client", "description": "Calcula a maior variação de bitrate para um cliente específico em intervalos de 5 minutos."},
                {"name": "calculate_min_latency_for_client", "description": "Calcula a menor latência registrada para um cliente específico."},
                {"name": "calculate_bitrate_variation_for_server", "description": "Calcula a variação do bitrate médio ao longo do dia para um servidor específico."}
            ],
            max_tokens=500,
            temperature=0.4
        )

        # Decodificar a função sugerida pelo GPT
        suggested_function = response.choices[0].message.function_call.name
        client_name = None
        server_name = None

        # Extração do nome do cliente ou servidor, se aplicável
        if 'cliente' in question:
            client_name = question.split("cliente")[-1].strip().lower().replace("?", "")
        elif 'servidor' in question:
            server_name = question.split("servidor")[-1].strip().lower().replace("?", "")

        # Adicionar uma verificação manual para perguntas específicas
        if "tempo médio de latência" in question and client_name:
            if client_name in self.latency_data['client'].str.lower().unique():
                return self.calculate_avg_latency_for_client(client_name), "Pergunta processada com sucesso."
            else:
                return f"Cliente {client_name} não encontrado nos dados.", "Pergunta processada com sucesso."
        elif suggested_function == 'calculate_avg_latency_for_client' and client_name:
            if client_name in self.latency_data['client'].str.lower().unique():
                return self.calculate_avg_latency_for_client(client_name), "Pergunta processada com sucesso."
            else:
                return f"Cliente {client_name} não encontrado nos dados.", "Pergunta processada com sucesso."
        # Continue com as verificações para outras funções
        elif suggested_function == 'calculate_qoe_with_explanation':
            return self.calculate_qoe_with_explanation(), "Pergunta processada com sucesso."
        elif suggested_function == 'calculate_most_consistent_server':
            return self.calculate_most_consistent_server(), "Pergunta processada com sucesso."
        elif suggested_function == 'calculate_best_server_switch_strategy' and client_name:
            if client_name in self.preprocessed_data['client'].unique():
                return self.calculate_best_server_switch_strategy(client_name), "Pergunta processada com sucesso."
            else:
                return f"Cliente {client_name} não encontrado nos dados.", "Pergunta processada com sucesso."
        elif suggested_function == 'calculate_qoe_with_increased_latency' and client_name:
            if client_name in self.preprocessed_data['client'].unique():
                return self.calculate_qoe_with_increased_latency(client_name), "Pergunta processada com sucesso."
            else:
                return f"Cliente {client_name} não encontrado nos dados.", "Pergunta processada com sucesso."
        elif suggested_function == 'calculate_avg_bitrate_for_server' and server_name:
            if server_name in self.bitrate_data['server'].str.lower().unique():
                return self.calculate_avg_bitrate_for_server(server_name), "Pergunta processada com sucesso."
            else:
                return f"Servidor {server_name} não encontrado nos dados.", "Pergunta processada com sucesso."
        elif suggested_function == 'calculate_max_bitrate_variation_for_client' and client_name:
            if client_name in self.bitrate_data['client'].str.lower().unique():
                return self.calculate_max_bitrate_variation_for_client(client_name), "Pergunta processada com sucesso."
            else:
                return f"Cliente {client_name} não encontrado nos dados.", "Pergunta processada com sucesso."
        elif suggested_function == 'calculate_min_latency_for_client' and client_name:
            if client_name in self.latency_data['client'].str.lower().unique():
                return self.calculate_min_latency_for_client(client_name), "Pergunta processada com sucesso."
            else:
                return f"Cliente {client_name} não encontrado nos dados.", "Pergunta processada com sucesso."
        elif suggested_function == 'calculate_bitrate_variation_for_server' and server_name:
            if server_name in self.bitrate_data['server'].str.lower().unique():
                return self.calculate_bitrate_variation_for_server(server_name), "Pergunta processada com sucesso."
            else:
                return f"Servidor {server_name} não encontrado nos dados.", "Pergunta processada com sucesso."
        else:
            return "Desculpe, não consegui identificar a ação correta a ser tomada.", "Pergunta não reconhecida."




    def calculate_qoe_with_explanation(self):
        # Calcular a QoE para cada cliente e determinar o cliente com a menor média
        grouped_data = self.preprocessed_data.groupby('client')['QoE'].mean()
        worst_client = grouped_data.idxmin()
        worst_client_qoe = grouped_data.min()

        # Informação principal
        main_info = f"O cliente com a pior qualidade de recepção de vídeo é: {worst_client}."

        # Solicitar uma explicação ao GPT
        explanation_prompt = (
            f"A qualidade de experiência (QoE) foi calculada para cada cliente como a razão "
            f"entre a taxa de transmissão normalizada e a latência normalizada. O cliente {worst_client} "
            f"teve a menor média de QoE ({worst_client_qoe:.4f}) ao longo do tempo. Explique de forma detalhada."
        )
        explanation = self.generate_gpt_explanation(explanation_prompt)
        return f"{main_info} {explanation}"

    def calculate_most_consistent_server(self):
        # Calcular a variância da QoE para cada servidor e identificar o servidor mais consistente
        server_variance = self.preprocessed_data.groupby('server')['QoE'].var()
        most_consistent_server = server_variance.idxmin()
        lowest_variance = server_variance.min()

        # Informação principal
        main_info = f"O servidor que fornece a QoE mais consistente é: {most_consistent_server}."

        # Solicitar uma explicação ao GPT
        explanation_prompt = (
            f"A consistência foi determinada pela menor variância da QoE ao longo do tempo. "
            f"O servidor {most_consistent_server} apresentou a menor variância ({lowest_variance:.4f}). "
            f"Explique de forma detalhada."
        )
        explanation = self.generate_gpt_explanation(explanation_prompt)
        return f"{main_info} {explanation}"

    def calculate_best_server_switch_strategy(self, client_name):
        # Verificar se o cliente existe nos dados
        if client_name not in self.preprocessed_data['client'].unique():
            return f"Cliente {client_name} não encontrado nos dados."

        # Filtrar os dados para o cliente especificado
        client_data = self.preprocessed_data[self.preprocessed_data['client'] == client_name]

        # Encontrar o melhor servidor para cada timestamp com base na maior QoE
        best_servers = client_data.loc[client_data.groupby('minute')['QoE'].idxmax()]

        # Informação principal
        main_info = f"O melhor plano de troca de servidor para maximizar a QoE do cliente {client_name} foi identificado."

        # Solicitar uma explicação ao GPT
        explanation_prompt = (
            f"O plano de troca foi derivado identificando o servidor com a melhor qualidade de experiência (QoE) para cada minuto, "
            f"permitindo otimizar a qualidade de recepção de vídeo para o cliente {client_name}. Explique de forma detalhada como "
            f"esse plano de troca maximiza a qualidade da experiência."
        )
        explanation = self.generate_gpt_explanation(explanation_prompt)
        return f"{main_info} {explanation}"

    def calculate_qoe_with_increased_latency(self, client_name):
        # Verificar se o cliente existe nos dados
        if client_name not in self.preprocessed_data['client'].unique():
            return f"Cliente {client_name} não encontrado nos dados."

        # Filtrar os dados para o cliente especificado
        client_data = self.preprocessed_data[self.preprocessed_data['client'] == client_name].copy()

        # Aumentar a latência em 20%
        client_data['Increased_Latency'] = client_data['rtt'] * 1.20

        # Normalizar a latência aumentada
        client_data['Normalized_Increased_Latency'] = self.normalize_data(client_data, 'Increased_Latency')

        # Recalcular a QoE com a latência aumentada
        client_data['New_QoE'] = client_data['Normalized_Bitrate'] / client_data['Normalized_Increased_Latency']

        # Informação principal
        main_info = f"A QoE do cliente {client_name} foi recalculada após aumentar a latência em 20%."

        # Solicitar uma explicação ao GPT
        explanation_prompt = (
            f"Após aumentar a latência do cliente {client_name} em 20%, a qualidade de experiência (QoE) foi recalculada. "
            f"Explique como essa mudança na latência impacta a QoE do cliente, destacando qualquer mudança significativa nos resultados."
        )
        explanation = self.generate_gpt_explanation(explanation_prompt)
        return f"{main_info} {explanation}"
    
    def calculate_avg_latency_for_client(self, client_name):
        # Verificar se o cliente existe nos dados
        if client_name not in self.latency_data['client'].unique():
            return f"Cliente {client_name} não encontrado nos dados."
        
        # Filtrar os dados para o cliente especificado
        client_latency_data = self.latency_data[self.latency_data['client'] == client_name]
        
        # Calcular a média da latência
        avg_latency = client_latency_data['latency'].mean()
        return f"O tempo médio de latência do cliente {client_name} é {avg_latency:.2f} ms."

    def calculate_avg_bitrate_for_server(self, server_name):
        # Verificar se o servidor existe nos dados
        if server_name not in self.bitrate_data['server'].unique():
            return f"Servidor {server_name} não encontrado nos dados."
        
        # Filtrar os dados para o servidor especificado
        server_bitrate_data = self.bitrate_data[self.bitrate_data['server'] == server_name]
        
        # Calcular a média do bitrate
        avg_bitrate = server_bitrate_data['bitrate'].mean()
        return f"O bitrate médio para o servidor {server_name} é {avg_bitrate:.2f} kbps."

    def calculate_max_bitrate_variation_for_client(self, client_name):
        # Verificar se o cliente existe nos dados
        if client_name not in self.bitrate_data['client'].unique():
            return f"Cliente {client_name} não encontrado nos dados."
        
        # Filtrar os dados para o cliente especificado
        client_bitrate_data = self.bitrate_data[self.bitrate_data['client'] == client_name]
        
        # Agrupar os dados por intervalos de 5 minutos e calcular a variação máxima
        client_bitrate_data['minute'] = pd.to_datetime(client_bitrate_data['timestamp'], unit='s').dt.floor('5min')
        grouped_bitrate = client_bitrate_data.groupby('minute')['bitrate'].apply(lambda x: x.max() - x.min())
        
        # Encontrar a maior variação
        max_variation = grouped_bitrate.max()
        return f"A maior variação de bitrate para o cliente {client_name} em um intervalo de 5 minutos é {max_variation:.2f} kbps."

    def calculate_min_latency_for_client(self, client_name):
        # Verificar se o cliente existe nos dados
        if client_name not in self.latency_data['client'].unique():
            return f"Cliente {client_name} não encontrado nos dados."
        
        # Filtrar os dados para o cliente especificado
        client_latency_data = self.latency_data[self.latency_data['client'] == client_name]
        
        # Encontrar a menor latência
        min_latency = client_latency_data['latency'].min()
        return f"A menor latência registrada para o cliente {client_name} é {min_latency:.2f} ms."

    def calculate_bitrate_variation_for_server(self, server_name):
        # Verificar se o servidor existe nos dados
        if server_name not in self.bitrate_data['server'].unique():
            return f"Servidor {server_name} não encontrado nos dados."
        
        # Filtrar os dados para o servidor especificado
        server_bitrate_data = self.bitrate_data[self.bitrate_data['server'] == server_name]
        
        # Converter timestamp para granularidade de hora
        server_bitrate_data['hour'] = pd.to_datetime(server_bitrate_data['timestamp'], unit='s').dt.hour
        
        # Agrupar por hora e calcular a média do bitrate
        hourly_bitrate_avg = server_bitrate_data.groupby('hour')['bitrate'].mean()
        
        # Resumir as variações ao longo do dia
        return f"As variações de bitrate médio ao longo do dia para o servidor {server_name} são: {hourly_bitrate_avg.to_dict()}."

    def generate_gpt_explanation(self, prompt):
        # Usar a API da OpenAI para gerar uma explicação detalhada, limitando a 300 tokens
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em análise de qualidade de experiência (QoE)."},
                {"role": "system", "content": "Responda de forma clara, usando no máximo 300 tokens."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    def run(self):
        # Executa a aplicação Dash
        self.app.run_server(debug=True)

# Execução da aplicação
if __name__ == '__main__':
    api_key = '****************************************************'
    app_instance = Main(api_key)
    app_instance.run()
