import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

def perform_time_series_analysis(df, date_column, value_column):
    """
    時系列データの分析を行う関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        分析するデータフレーム
    date_column : str
        日付列の名前
    value_column : str
        分析対象の値の列名
        
    Returns:
    --------
    dict
        分析結果と可視化用のグラフオブジェクト
    """
    # 日付でソート
    df = df.sort_values(by=date_column)
    
    # 日付列をインデックスに設定
    time_series_df = df.set_index(date_column)[[value_column]].copy()
    
    # 結果を格納する辞書
    results = {}
    
    # 移動平均の計算
    windows = [7, 14, 30]
    for window in windows:
        if len(time_series_df) > window:
            time_series_df[f'{value_column}_MA{window}'] = time_series_df[value_column].rolling(window=window).mean()
    
    # トレンドと季節性の分析
    if len(time_series_df) >= 30:  # 十分なデータがある場合
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # 頻度を推定
            if 'D' in time_series_df.index.freq or time_series_df.index.freq is None:
                freq = 'D'  # 日次データと仮定
                period = 7  # 週次の季節性を仮定
            else:
                freq = time_series_df.index.freq
                period = 12  # 月次の季節性を仮定（月次データの場合）
            
            # インデックスが日付型であることを確認
            if pd.api.types.is_datetime64_any_dtype(time_series_df.index):
                # データの欠損値を線形補間で埋める
                time_series_df_filled = time_series_df[[value_column]].fillna(method='ffill').fillna(method='bfill')
                
                # 季節分解を実行
                decomposition = seasonal_decompose(time_series_df_filled, model='additive', period=period)
                
                # 結果を辞書に格納
                results['trend'] = decomposition.trend
                results['seasonal'] = decomposition.seasonal
                results['residual'] = decomposition.resid
                
                # 季節分解のプロット
                fig = make_subplots(rows=4, cols=1, 
                                   subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])
                
                fig.add_trace(go.Scatter(x=time_series_df.index, y=time_series_df[value_column], 
                                        name='Observed'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, 
                                        name='Trend'), row=2, col=1)
                
                fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, 
                                        name='Seasonal'), row=3, col=1)
                
                fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, 
                                        name='Residual'), row=4, col=1)
                
                fig.update_layout(height=800, title_text=f"{value_column}の時系列分解")
                
                results['decomposition_plot'] = fig
        except Exception as e:
            results['error'] = f"時系列分解中にエラーが発生しました: {e}"
    
    # 自己相関分析
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import io
        from PIL import Image
        
        # 欠損値の処理
        ts_data = time_series_df[value_column].fillna(method='ffill').fillna(method='bfill')
        
        # 自己相関プロット
        acf_buf = io.BytesIO()
        fig_acf, ax = plt.subplots(figsize=(10, 4))
        plot_acf(ts_data, ax=ax, lags=min(30, len(ts_data) // 2))
        ax.set_title(f"{value_column}の自己相関関数")
        plt.tight_layout()
        fig_acf.savefig(acf_buf, format='png')
        plt.close(fig_acf)
        
        # 偏自己相関プロット
        pacf_buf = io.BytesIO()
        fig_pacf, ax = plt.subplots(figsize=(10, 4))
        plot_pacf(ts_data, ax=ax, lags=min(30, len(ts_data) // 2))
        ax.set_title(f"{value_column}の偏自己相関関数")
        plt.tight_layout()
        fig_pacf.savefig(pacf_buf, format='png')
        plt.close(fig_pacf)
        
        # 結果に保存
        acf_buf.seek(0)
        pacf_buf.seek(0)
        results['acf_plot'] = acf_buf
        results['pacf_plot'] = pacf_buf
    except Exception as e:
        results['error'] = f"自己相関分析中にエラーが発生しました: {e}"
    
    # 基本的な時系列プロット
    fig = px.line(df, x=date_column, y=value_column, title=f"{value_column}の時系列データ")
    
    # 移動平均線を追加
    for window in windows:
        ma_col = f'{value_column}_MA{window}'
        if ma_col in time_series_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=time_series_df.index,
                    y=time_series_df[ma_col],
                    mode='lines',
                    name=f'{window}日移動平均'
                )
            )
    
    results['time_series_plot'] = fig
    
    return results


def perform_correlation_analysis(df, numeric_columns):
    """
    相関分析を行う関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        分析するデータフレーム
    numeric_columns : list
        分析対象の数値列のリスト
        
    Returns:
    --------
    dict
        分析結果と可視化用のグラフオブジェクト
    """
    # 結果を格納する辞書
    results = {}
    
    # 数値列のみを抽出
    numeric_df = df[numeric_columns].copy()
    
    # 相関係数行列の計算
    corr_matrix = numeric_df.corr()
    results['correlation_matrix'] = corr_matrix
    
    # ヒートマップの生成
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="相関係数ヒートマップ"
    )
    results['heatmap'] = fig
    
    # 強い相関を持つ変数ペアの特定
    strong_correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) > 0.5:  # 相関係数の絶対値が0.5より大きいペアを抽出
                strong_correlations.append({
                    'variable1': col1,
                    'variable2': col2,
                    'correlation': corr_value
                })
    
    # 強い相関を持つペアを相関係数の絶対値でソート
    strong_correlations = sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)
    results['strong_correlations'] = strong_correlations
    
    # 強い相関を持つ上位のペアの散布図
    scatter_plots = []
    
    for i, corr_pair in enumerate(strong_correlations[:min(5, len(strong_correlations))]):
        col1 = corr_pair['variable1']
        col2 = corr_pair['variable2']
        corr = corr_pair['correlation']
        
        fig = px.scatter(
            df, x=col1, y=col2, 
            trendline="ols",
            title=f"{col1} vs {col2} (相関係数: {corr:.3f})"
        )
        scatter_plots.append(fig)
    
    results['scatter_plots'] = scatter_plots
    
    return results


def perform_cluster_analysis(df, numeric_columns, n_clusters=3):
    """
    クラスター分析を行う関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        分析するデータフレーム
    numeric_columns : list
        分析に使用する数値列のリスト
    n_clusters : int
        クラスターの数
        
    Returns:
    --------
    dict
        分析結果と可視化用のグラフオブジェクト
    """
    # 結果を格納する辞書
    results = {}
    
    # 数値列のみを抽出
    numeric_df = df[numeric_columns].copy()
    
    # 欠損値の処理
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    # スケーリング
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # KMeansクラスタリングの実行
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # クラスターラベルをデータフレームに追加
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    results['df_with_clusters'] = df_with_clusters
    
    # クラスターごとの基本統計量
    cluster_stats = df_with_clusters.groupby('cluster')[numeric_columns].agg(['mean', 'std', 'min', 'max'])
    results['cluster_stats'] = cluster_stats
    
    # 次元削減（PCA）
    if len(numeric_columns) > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'cluster': clusters
        })
        
        # PCAの寄与率
        explained_variance = pca.explained_variance_ratio_
        results['explained_variance'] = explained_variance
        
        # PCAの散布図
        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='cluster',
            title=f"PCAによる2次元プロット (説明率: {explained_variance[0]:.2f}, {explained_variance[1]:.2f})",
            labels={'PC1': f'主成分1 ({explained_variance[0]:.2f})', 'PC2': f'主成分2 ({explained_variance[1]:.2f})'},
            color_continuous_scale='viridis'
        )
        results['pca_plot'] = fig
        
        # 主成分の寄与度
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=numeric_columns
        )
        results['pca_loadings'] = loadings
        
        # 変数寄与度の可視化
        fig_loadings = px.bar(
            loadings.reset_index(),
            x='index', y=['PC1', 'PC2'],
            barmode='group',
            title="各変数の主成分への寄与度",
            labels={'index': '変数', 'value': '寄与度'}
        )
        results['loadings_plot'] = fig_loadings
    
    # クラスター別の箱ひげ図（最も特徴的な変数を選択）
    if len(numeric_columns) > 0:
        # Kruskal-Wallis検定でクラスター間で最も差がある変数を特定
        from scipy.stats import kruskal
        
        kruskal_results = {}
        for col in numeric_columns:
            try:
                groups = [df_with_clusters[df_with_clusters['cluster'] == c][col].dropna() for c in range(n_clusters)]
                
                # 各グループに十分なデータがあるか確認
                if all(len(g) > 5 for g in groups):
                    stat, p = kruskal(*groups)
                    kruskal_results[col] = {'statistic': stat, 'p-value': p}
            except:
                continue
        
        if kruskal_results:
            # p値でソート
            sorted_vars = sorted(kruskal_results.items(), key=lambda x: x[1]['p-value'])
            top_vars = [item[0] for item in sorted_vars[:min(5, len(sorted_vars))]]
        else:
            # 統計的検定ができない場合は最初の5つの変数を使用
            top_vars = numeric_columns[:min(5, len(numeric_columns))]
        
        results['top_discriminating_vars'] = top_vars
        
        # 箱ひげ図
        boxplots = []
        for var in top_vars:
            fig = px.box(
                df_with_clusters, x='cluster', y=var,
                title=f"クラスター別の {var} 分布"
            )
            boxplots.append(fig)
        
        results['boxplots'] = boxplots
    
    return results


def perform_distribution_analysis(df, column):
    """
    列の分布分析を行う関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        分析するデータフレーム
    column : str
        分析対象の列名
        
    Returns:
    --------
    dict
        分析結果と可視化用のグラフオブジェクト
    """
    # 結果を格納する辞書
    results = {}
    
    # 基本統計量
    if pd.api.types.is_numeric_dtype(df[column]):
        stats = df[column].describe()
        results['stats'] = stats
        
        # 歪度と尖度
        from scipy.stats import skew, kurtosis
        
        results['skewness'] = skew(df[column].dropna())
        results['kurtosis'] = kurtosis(df[column].dropna())
        
        # ヒストグラムとKDEプロット
        fig = px.histogram(
            df, x=column,
            histnorm='probability density',
            title=f"{column}の分布",
            marginal="box"
        )
        
        results['histogram'] = fig
        
        # 正規性検定
        from scipy.stats import shapiro, normaltest
        
        # サンプルサイズが大きすぎる場合は小さいサブサンプルで検定
        sample = df[column].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        
        if len(sample) >= 20:  # 十分なサンプルサイズがある場合のみ検定
            try:
                # Shapiro-Wilk検定（より小さいサンプル向け）
                if len(sample) <= 5000:
                    w, p_shapiro = shapiro(sample)
                    results['shapiro_test'] = {'statistic': w, 'p-value': p_shapiro}
                
                # D'Agostino's K^2検定（大きいサンプル向け）
                k2, p_normal = normaltest(sample)
                results['normaltest'] = {'statistic': k2, 'p-value': p_normal}
            except:
                pass
        
        # QQプロット
        from scipy.stats import probplot
        import io
        from PIL import Image
        
        # QQプロットの作成
        qq_buf = io.BytesIO()
        fig_qq, ax = plt.subplots(figsize=(8, 6))
        probplot(df[column].dropna(), plot=ax)
        ax.set_title(f"{column}のQQプロット")
        plt.tight_layout()
        fig_qq.savefig(qq_buf, format='png')
        plt.close(fig_qq)
        
        # 結果に保存
        qq_buf.seek(0)
        results['qq_plot'] = qq_buf
        
    else:  # カテゴリ変数の場合
        # 頻度カウント
        value_counts = df[column].value_counts()
        results['value_counts'] = value_counts
        
        # 棒グラフ
        fig = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            title=f"{column}の値の分布",
            labels={'x': column, 'y': '頻度'},
        )
        
        results['bar_chart'] = fig
    
    return results