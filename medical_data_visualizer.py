import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importar os dados
df = pd.read_csv('boilerplate-medical-data-visualizer/medical_examination.csv')

# 1. Adicionar a coluna 'overweight'
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 2. Normalizar os dados de colesterol e glicose
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Função para desenhar o gráfico categórico
def draw_cat_plot():
    # 3. Criar DataFrame para gráfico
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 4. Agrupar e reformular dados
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 5. Desenhar gráfico com seaborn
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig
    fig.savefig('catplot.png')
    return fig

# Função para desenhar o Mapa de Calor
def draw_heat_map():
    # 6. Limpar os dados
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    # 7. Calcular a matriz de correlação
    corr = df_heat.corr()

    # 8. Gerar uma máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 9. Configurar a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 10. Desenhar o Mapa de Calor
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', ax=ax, cmap='coolwarm', square=True, linewidths=0.5)

    # Salvar figura e retornar
    fig.savefig('heatmap.png')
    return fig
