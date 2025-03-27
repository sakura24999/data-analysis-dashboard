import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
import io

# 自作モジュールのインポート
# 注: 実行時にはimport errorを防ぐため、同じディレクトリに配置してください
from data_preprocessing import preprocess_data
from advanced_analysis import (
    perform_time_series_analysis,
    perform_correlation_analysis,
    perform_cluster_analysis,
    perform_distribution_analysis
)

# ページ設定
st.set_page_config(
    page_title="データ分析ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'preprocessing_config' not in st.session_state:
    st.session_state.preprocessing_config = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# ダッシュボードのタイトル
st.title("データ分析ダッシュボード")
st.markdown("### データの可視化・分析・前処理を一括で行えるアプリケーション")

# サイドバーの設定
st.sidebar.title("メニュー")
app_mode = st.sidebar.selectbox(
    "モードを選択",
    ["データ読み込み", "データ探索", "データ前処理", "高度な分析", "レポート生成"]
)

# --------------------------------
# データ読み込み機能
# --------------------------------
if app_mode == "データ読み込み":
    st.header("データ読み込み")
    
    # データソースの選択
    data_source = st.radio(
        "データソースを選択",
        ["サンプルデータ", "CSVファイルをアップロード", "Excelファイルをアップロード"]
    )
    
    if data_source == "サンプルデータ":
        # サンプルデータの種類を選択
        sample_data_type = st.selectbox(
            "サンプルデータの種類",
            ["売上データ", "株価データ", "気象データ"]
        )
        
        # サンプルデータの生成
        if sample_data_type == "売上データ":
            # 日付範囲の生成
            date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            
            # 売上データの生成
            np.random.seed(42)  # 再現性のための固定シード
            sales_data = {
                '日付': date_range,
                '売上': np.random.normal(1000, 200, len(date_range)),
                '商品A': np.random.normal(500, 100, len(date_range)),
                '商品B': np.random.normal(300, 80, len(date_range)),
                '商品C': np.random.normal(200, 50, len(date_range)),
            }
            
            # 季節性を追加
            for i, date in enumerate(date_range):
                # 週末は売上増加
                if date.dayofweek >= 5:  # 5:土曜日, 6:日曜日
                    sales_data['売上'][i] *= 1.5
                    sales_data['商品A'][i] *= 1.3
                    sales_data['商品B'][i] *= 1.7
                    sales_data['商品C'][i] *= 1.4
                
                # 月の初めは売上増加
                if date.day <= 5:
                    sales_data['売上'][i] *= 1.2
                
                # 季節トレンド（夏と冬に売上増加）
                month = date.month
                if month in [6, 7, 8]:  # 夏
                    sales_data['売上'][i] *= 1.1
                    sales_data['商品A'][i] *= 1.3
                elif month in [11, 12, 1]:  # 冬
                    sales_data['売上'][i] *= 1.2
                    sales_data['商品B'][i] *= 1.4
            
            df = pd.DataFrame(sales_data)
            
        elif sample_data_type == "株価データ":
            # 日付範囲（営業日のみ）
            date_range = pd.bdate_range(start='2023-01-01', end='2023-12-31')
            
            np.random.seed(42)  # 再現性のための固定シード
            
            # 初期株価
            initial_price = 1000
            
            # ランダムウォークで株価を生成
            price_changes = np.random.normal(0.0005, 0.015, len(date_range))
            prices = [initial_price]
            
            for change in price_changes:
                prices.append(prices[-1] * (1 + change))
            
            prices = prices[1:]  # 最初の要素を削除
            
            # ボリューム（取引量）を生成
            volume = np.random.normal(1000000, 200000, len(date_range))
            
            # データフレーム作成
            stock_data = {
                '日付': date_range,
                '始値': prices * np.random.normal(0.995, 0.002, len(prices)),
                '高値': prices * np.random.normal(1.01, 0.003, len(prices)),
                '安値': prices * np.random.normal(0.99, 0.003, len(prices)),
                '終値': prices,
                '出来高': volume
            }
            
            df = pd.DataFrame(stock_data)
            
            # 一貫性のある価格データにする
            for i in range(len(df)):
                high = max(df.loc[i, '始値'], df.loc[i, '終値']) * np.random.uniform(1.001, 1.015)
                low = min(df.loc[i, '始値'], df.loc[i, '終値']) * np.random.uniform(0.985, 0.999)
                df.loc[i, '高値'] = high
                df.loc[i, '安値'] = low
            
        elif sample_data_type == "気象データ":
            # 日付範囲の生成
            date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            
            np.random.seed(42)  # 再現性のための固定シード
            
            # 気温データ生成（季節性を持たせる）
            temp_base = 15  # 平均気温の基準値
            temp_amplitude = 10  # 年間の気温振幅
            
            temperatures = []
            humidity = []
            precipitation = []
            wind_speed = []
            
            for date in date_range:
                # 日付から年間の位置（0〜1）を計算
                day_of_year = date.dayofyear
                year_progress = day_of_year / 365.25
                
                # 季節性を持つ気温を生成（夏に最高、冬に最低）
                seasonal_component = temp_amplitude * np.sin(2 * np.pi * (year_progress - 0.25))
                daily_variation = np.random.normal(0, 2)  # 日々のランダム変動
                temp = temp_base + seasonal_component + daily_variation
                temperatures.append(temp)
                
                # 湿度（気温と逆相関）
                base_humidity = 70 - seasonal_component  # 夏は乾燥、冬は湿度高め
                humidity_variation = np.random.normal(0, 5)
                hum = max(min(base_humidity + humidity_variation, 100), 10)  # 10%〜100%に制限
                humidity.append(hum)
                
                # 降水量（確率的に発生、湿度が高いほど確率上昇）
                rain_prob = hum / 100  # 湿度を降水確率として使用
                if np.random.random() < rain_prob * 0.3:  # 降水確率を調整
                    rain_amount = np.random.exponential(5)  # 指数分布で降水量を生成
                else:
                    rain_amount = 0
                precipitation.append(rain_amount)
                
                # 風速
                wind = np.random.gamma(2, 1.5)  # ガンマ分布で風速を生成
                wind_speed.append(wind)
            
            weather_data = {
                '日付': date_range,
                '気温(℃)': temperatures,
                '湿度(%)': humidity,
                '降水量(mm)': precipitation,
                '風速(m/s)': wind_speed
            }
            
            df = pd.DataFrame(weather_data)
        
        # データフレームの表示
        st.subheader("データプレビュー")
        st.dataframe(df.head(10))
        
        # データの詳細情報を表示
        st.subheader("データセット情報")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"行数: {df.shape[0]}")
            st.write(f"列数: {df.shape[1]}")
        with col2:
            st.write(f"欠損値の数: {df.isnull().sum().sum()}")
            st.write(f"メモリ使用量: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # データの保存
        if st.button("このデータセットを使用"):
            st.session_state.data = df
            st.session_state.processed_data = df.copy()
            st.success("データセットを読み込みました。左のサイドバーから「データ探索」を選択して分析を開始できます。")
    
    elif data_source == "CSVファイルをアップロード":
        st.subheader("CSVファイルのアップロード")
        
        uploaded_file = st.file_uploader("CSVファイルを選択してください", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # エンコーディングの選択
                encoding = st.selectbox(
                    "ファイルのエンコーディング",
                    ["utf-8", "shift-jis", "cp932", "latin1"],
                    index=0
                )
                
                # 区切り文字の選択
                delimiter = st.selectbox(
                    "区切り文字",
                    [",", ";", "\t"],
                    index=0,
                    format_func=lambda x: "カンマ (,)" if x == "," else "セミコロン (;)" if x == ";" else "タブ (\\t)"
                )
                
                # CSVファイルの読み込み
                df = pd.read_csv(uploaded_file, encoding=encoding, delimiter=delimiter)
                
                # データフレームの表示
                st.subheader("データプレビュー")
                st.dataframe(df.head(10))
                
                # データの詳細情報を表示
                st.subheader("データセット情報")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"行数: {df.shape[0]}")
                    st.write(f"列数: {df.shape[1]}")
                with col2:
                    st.write(f"欠損値の数: {df.isnull().sum().sum()}")
                    st.write(f"メモリ使用量: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                # データの保存
                if st.button("このデータセットを使用"):
                    st.session_state.data = df
                    st.session_state.processed_data = df.copy()
                    st.success("データセットを読み込みました。左のサイドバーから「データ探索」を選択して分析を開始できます。")
            
            except Exception as e:
                st.error(f"ファイルの読み込みに失敗しました: {e}")
    
    elif data_source == "Excelファイルをアップロード":
        st.subheader("Excelファイルのアップロード")
        
        uploaded_file = st.file_uploader("Excelファイルを選択してください", type=["xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # シートの選択
                xls = pd.ExcelFile(uploaded_file)
                sheet_name = st.selectbox(
                    "シートを選択",
                    xls.sheet_names
                )
                
                # ファイルの読み込み
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                
                # データフレームの表示
                st.subheader("データプレビュー")
                st.dataframe(df.head(10))
                
                # データの詳細情報を表示
                st.subheader("データセット情報")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"行数: {df.shape[0]}")
                    st.write(f"列数: {df.shape[1]}")
                with col2:
                    st.write(f"欠損値の数: {df.isnull().sum().sum()}")
                    st.write(f"メモリ使用量: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                # データの保存
                if st.button("このデータセットを使用"):
                    st.session_state.data = df
                    st.session_state.processed_data = df.copy()
                    st.success("データセットを読み込みました。左のサイドバーから「データ探索」を選択して分析を開始できます。")
            
            except Exception as e:
                st.error(f"ファイルの読み込みに失敗しました: {e}")

# --------------------------------
# データ探索機能
# --------------------------------
elif app_mode == "データ探索":
    st.header("データ探索")
    
    if st.session_state.data is None:
        st.warning("まずデータを読み込んでください。左のサイドバーから「データ読み込み」を選択してデータを読み込みます。")
    else:
        # データフレームの取得
        df = st.session_state.processed_data
        
        # タブの設定
        explore_tabs = st.tabs(["データプレビュー", "統計要約", "可視化", "相関分析"])
        
        # データプレビュータブ
        with explore_tabs[0]:
            st.subheader("データプレビュー")
            
            # 表示する行数の選択
            n_rows = st.slider("表示する行数", 5, 100, 10)
            
            # データフレームの表示
            st.dataframe(df.head(n_rows))
            
            # データ型情報の表示
            st.subheader("データ型情報")
            
            # 列ごとのデータ型とデータ例を表形式で表示
            dtype_df = pd.DataFrame({
                'データ型': df.dtypes,
                '非欠損値数': df.count(),
                '欠損値数': df.isnull().sum(),
                '欠損率(%)': (df.isnull().sum() / len(df) * 100).round(2),
                'ユニーク値数': df.nunique(),
                'サンプル値': [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else '' for col in df.columns]
            })
            
            st.dataframe(dtype_df)
        
        # 統計要約タブ
        with explore_tabs[1]:
            st.subheader("統計要約")
            
            # 数値列と非数値列を分ける
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            
            # 数値列の統計量
            if numeric_cols:
                st.subheader("数値列の統計量")
                st.dataframe(df[numeric_cols].describe().T)
            
            # 非数値列の統計量
            if non_numeric_cols:
                st.subheader("非数値列の情報")
                for col in non_numeric_cols:
                    st.write(f"**{col}** のトップ値:")
                    try:
                        st.dataframe(df[col].value_counts().head(5))
                    except:
                        st.write("この列の集計に失敗しました。")
        
        # 可視化タブ
        with explore_tabs[2]:
            st.subheader("データ可視化")
            
            # グラフの種類を選択
            chart_type = st.selectbox(
                "グラフの種類",
                ["折れ線グラフ", "棒グラフ", "散布図", "ヒストグラム", "箱ひげ図", "パイチャート", "ヒートマップ"]
            )
            
            if chart_type == "折れ線グラフ":
                # X軸として使用する列
                x_column = st.selectbox(
                    "X軸（時間軸）の列を選択",
                    df.columns.tolist()
                )
                
                # Y軸として使用する列
                y_columns = st.multiselect(
                    "Y軸の列を選択（複数選択可）",
                    numeric_cols
                )
                
                if x_column and y_columns:
                    try:
                        # X軸を日付型に変換してみる
                        try:
                            x_data = pd.to_datetime(df[x_column])
                        except:
                            x_data = df[x_column]
                        
                        fig = px.line(df, x=x_column, y=y_columns, title="時系列データ")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"グラフの作成に失敗しました: {e}")
            
            elif chart_type == "棒グラフ":
                # X軸として使用する列
                x_column = st.selectbox(
                    "X軸（カテゴリ）の列を選択",
                    df.columns.tolist()
                )
                
                # Y軸として使用する列
                y_column = st.selectbox(
                    "Y軸（数値）の列を選択",
                    numeric_cols if numeric_cols else ["なし"]
                )
                
                if x_column and y_column != "なし":
                    # カテゴリ数が多すぎる場合は上位N件に制限
                    max_categories = st.slider("表示するカテゴリ数", 5, 30, 10)
                    
                    try:
                        # 集計方法を選択
                        agg_method = st.selectbox(
                            "集計方法",
                            ["合計", "平均", "中央値", "最大値", "最小値"]
                        )
                        
                        # 集計方法に応じてデータを集計
                        agg_func = {
                            "合計": "sum",
                            "平均": "mean",
                            "中央値": "median",
                            "最大値": "max",
                            "最小値": "min"
                        }[agg_method]
                        
                        # データを集計
                        agg_data = df.groupby(x_column)[y_column].agg(agg_func).sort_values(ascending=False)
                        
                        # トップNのカテゴリを選択
                        agg_data = agg_data.head(max_categories)
                        
                        fig = px.bar(
                            x=agg_data.index, 
                            y=agg_data.values,
                            title=f"{x_column}ごとの{y_column}の{agg_method}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"グラフの作成に失敗しました: {e}")
            
            elif chart_type == "散布図":
                # X軸として使用する列
                x_column = st.selectbox(
                    "X軸の列を選択",
                    numeric_cols if numeric_cols else ["なし"]
                )
                
                # Y軸として使用する列
                y_column = st.selectbox(
                    "Y軸の列を選択",
                    numeric_cols if numeric_cols else ["なし"],
                    index=min(1, len(numeric_cols)-1) if len(numeric_cols) > 1 else 0
                )
                
                # 色分け用の列（オプション）
                color_column = st.selectbox(
                    "色分け用の列（オプション）",
                    ["なし"] + df.columns.tolist()
                )
                
                color_column = None if color_column == "なし" else color_column
                
                if x_column != "なし" and y_column != "なし":
                    try:
                        fig = px.scatter(
                            df, 
                            x=x_column, 
                            y=y_column, 
                            color=color_column,
                            title=f"{x_column} vs {y_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 相関係数の表示
                        correlation = df[x_column].corr(df[y_column])
                        st.metric("相関係数", f"{correlation:.4f}")
                    except Exception as e:
                        st.error(f"グラフの作成に失敗しました: {e}")
            
            elif chart_type == "ヒストグラム":
                # 列を選択
                column = st.selectbox(
                    "列を選択",
                    numeric_cols if numeric_cols else ["なし"]
                )
                
                if column != "なし":
                    try:
                        # ビンの数を設定
                        bin_count = st.slider("ビン（区間）の数", 5, 100, 20)
                        
                        # 正規化するかどうか
                        normalize = st.checkbox("確率密度に正規化", value=False)
                        
                        # ヒストグラムを描画
                        fig = px.histogram(
                            df, x=column, 
                            nbins=bin_count, 
                            title=f"{column}のヒストグラム",
                            histnorm='probability density' if normalize else None,
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 基本統計量
                        stats = df[column].describe()
                        st.write(f"平均値: {stats['mean']:.2f}, 標準偏差: {stats['std']:.2f}")
                        st.write(f"最小値: {stats['min']:.2f}, 最大値: {stats['max']:.2f}")
                        st.write(f"第1四分位数: {stats['25%']:.2f}, 中央値: {stats['50%']:.2f}, 第3四分位数: {stats['75%']:.2f}")
                    except Exception as e:
                        st.error(f"グラフの作成に失敗しました: {e}")
            
            elif chart_type == "箱ひげ図":
                # Y軸として使用する列
                y_column = st.selectbox(
                    "数値列を選択",
                    numeric_cols if numeric_cols else ["なし"]
                )
                
                # グループ化の列（オプション）
                x_column = st.selectbox(
                    "グループ化する列（オプション）",
                    ["なし"] + df.columns.tolist()
                )
                
                x_column = None if x_column == "なし" else x_column
                
                if y_column != "なし":
                    try:
                        if x_column:
                            # カテゴリ数が多すぎる場合は上位N件に制限
                            categories = df[x_column].value_counts().index
                            max_categories = st.slider("表示するカテゴリ数", 2, 30, min(10, len(categories)))
                            top_categories = categories[:max_categories]
                            
                            boxplot_data = df[df[x_column].isin(top_categories)]
                        else:
                            boxplot_data = df
                        
                        fig = px.box(
                            boxplot_data, 
                            x=x_column, 
                            y=y_column,
                            title=f"{y_column}の分布" + (f"（{x_column}ごと）" if x_column else "")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"グラフの作成に失敗しました: {e}")
            
            elif chart_type == "パイチャート":
                # カテゴリ列を選択
                column = st.selectbox(
                    "カテゴリ列を選択",
                    df.columns.tolist()
                )
                
                if column:
                    try:
                        # カテゴリ数が多すぎる場合は上位N件に制限し、残りは「その他」にまとめる
                        max_categories = st.slider("表示するカテゴリ数", 2, 20, 5)
                        
                        # カテゴリごとの集計
                        counts = df[column].value_counts()
                        
                        # トップNとその他に分類
                        if len(counts) > max_categories:
                            top_counts = counts.iloc[:max_categories]
                            other_count = counts.iloc[max_categories:].sum()
                            
                            labels = list(top_counts.index) + ['その他']
                            values = list(top_counts.values) + [other_count]
                        else:
                            labels = counts.index
                            values = counts.values
                        
                        fig = px.pie(
                            names=labels, 
                            values=values,
                            title=f"{column}の分布"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"グラフの作成に失敗しました: {e}")
            
            elif chart_type == "ヒートマップ":
                if len(numeric_cols) < 2:
                    st.warning("ヒートマップを作成するには、少なくとも2つの数値列が必要です。")
                else:
                    try:
                        # 相関行列の計算
                        corr_matrix = df[numeric_cols].corr()
                        
                        # ヒートマップの作成
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="相関係数ヒートマップ"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"グラフの作成に失敗しました: {e}")
        
        # 相関分析タブ
        with explore_tabs[3]:
            st.subheader("相関分析")
            
            # 数値列の選択
            if len(numeric_cols) < 2:
                st.warning("相関分析を行うには、少なくとも2つの数値列が必要です。")
            else:
                # 相関分析を実行
                corr_results = perform_correlation_analysis(df, numeric_cols)
                
                # 相関行列ヒートマップの表示
                st.subheader("相関係数ヒートマップ")
                st.plotly_chart(corr_results['heatmap'], use_container_width=True)
                
                # 強い相関を持つ変数ペアの表示
                st.subheader("強い相関を持つ変数ペア")
                if corr_results['strong_correlations']:
                    strong_corr_df = pd.DataFrame(corr_results['strong_correlations'])
                    st.dataframe(strong_corr_df)
                    
                    # 散布図の表示
                    st.subheader("強い相関を持つ変数ペアの散布図")
                    for i, fig in enumerate(corr_results['scatter_plots']):
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("相関係数が0.5を超える強い相関関係のある変数ペアが見つかりませんでした。")

# --------------------------------
# データ前処理機能
# --------------------------------
elif app_mode == "データ前処理":
    st.header("データ前処理")
    
    if st.session_state.data is None:
        st.warning("まずデータを読み込んでください。左のサイドバーから「データ読み込み」を選択してデータを読み込みます。")
    else:
        # 元のデータフレームとデータ前処理後のデータフレーム
        original_df = st.session_state.data
        processed_df = st.session_state.processed_data
        
        # タブの設定
        preprocess_tabs = st.tabs(["列の変更", "欠損値処理", "外れ値処理", "スケーリング", "特徴量エンジニアリング", "処理結果"])
        
        # 列の変更タブ
        with preprocess_tabs[0]:
            st.subheader("列の選択と名前変更")
            
            # 列の選択
            all_columns = original_df.columns.tolist()
            selected_columns = st.multiselect(
                "保持する列を選択",
                all_columns,
                default=all_columns
            )
            
            # 選択した列のみのデータフレーム
            if selected_columns:
                processed_df = processed_df[selected_columns]
                
                # 列の名前変更
                st.subheader("列名の変更（オプション）")
                
                rename_columns = {}
                for col in selected_columns:
                    new_name = st.text_input(f"{col} の新しい名前", value=col)
                    if new_name != col:
                        rename_columns[col] = new_name
                
                if rename_columns:
                    processed_df = processed_df.rename(columns=rename_columns)
                
                # 処理の適用
                if st.button("列の変更を適用"):
                    st.session_state.processed_data = processed_df
                    st.session_state.preprocessing_config['selected_columns'] = selected_columns
                    st.session_state.preprocessing_config['rename_columns'] = rename_columns
                    st.success("列の変更を適用しました。")
            else:
                st.warning("少なくとも1つの列を選択してください。")
        
        # 欠損値処理タブ
        with preprocess_tabs[1]:
            st.subheader("欠損値の処理")
            
            # 各列の欠損値情報
            missing_info = pd.DataFrame({
                '欠損値数': processed_df.isnull().sum(),
                '欠損率(%)': (processed_df.isnull().sum() / len(processed_df) * 100).round(2)
            }).sort_values('欠損値数', ascending=False)
            
            st.dataframe(missing_info)
            
            # 欠損値がある列のみ処理
            missing_columns = missing_info[missing_info['欠損値数'] > 0].index.tolist()
            
            if missing_columns:
                st.subheader("欠損値処理方法の選択")
                
                handle_missing = {}
                for col in missing_columns:
                    method = st.selectbox(
                        f"{col} の欠損値処理方法",
                        ["処理しない", "削除", "平均値", "中央値", "最頻値", "ゼロ", "前の値で埋める", "後の値で埋める"],
                        format_func=lambda x: {
                            "処理しない": "処理しない",
                            "削除": "行を削除",
                            "平均値": "平均値で埋める",
                            "中央値": "中央値で埋める",
                            "最頻値": "最頻値で埋める",
                            "ゼロ": "ゼロで埋める",
                            "前の値で埋める": "前の値で埋める (前方補完)",
                            "後の値で埋める": "後の値で埋める (後方補完)"
                        }[x]
                    )
                    
                    if method != "処理しない":
                        handle_missing[col] = {
                            "削除": "drop",
                            "平均値": "mean",
                            "中央値": "median",
                            "最頻値": "mode",
                            "ゼロ": "zero",
                            "前の値で埋める": "forward",
                            "後の値で埋める": "backward"
                        }[method]
                
                # 処理の適用
                if handle_missing and st.button("欠損値処理を適用"):
                    # 前処理設定の保存
                    st.session_state.preprocessing_config['handle_missing'] = handle_missing
                    
                    # データ前処理の実行
                    processed_df = preprocess_data(processed_df, {'handle_missing': handle_missing})
                    
                    # 処理後のデータを保存
                    st.session_state.processed_data = processed_df
                    
                    st.success("欠損値処理を適用しました。")
            else:
                st.info("データセットに欠損値はありません。")
        
        # 外れ値処理タブ
        with preprocess_tabs[2]:
            st.subheader("外れ値の処理")
            
            # 数値列のみ処理
            numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_cols:
                # 外れ値を可視化
                st.subheader("外れ値の可視化")
                
                # 可視化する列の選択
                viz_column = st.selectbox(
                    "可視化する列を選択",
                    numeric_cols
                )
                
                if viz_column:
                    # 箱ひげ図で外れ値を確認
                    fig = px.box(processed_df, y=viz_column, title=f"{viz_column}の箱ひげ図")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # IQRによる外れ値検出
                    Q1 = processed_df[viz_column].quantile(0.25)
                    Q3 = processed_df[viz_column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = processed_df[(processed_df[viz_column] < lower_bound) | 
                                            (processed_df[viz_column] > upper_bound)][viz_column]
                    
                    # 外れ値の情報を表示
                    st.write(f"**外れ値の統計:**")
                    st.write(f"- 下限しきい値: {lower_bound:.2f}")
                    st.write(f"- 上限しきい値: {upper_bound:.2f}")
                    st.write(f"- 外れ値の数: {len(outliers)}")
                    st.write(f"- 全体に対する割合: {len(outliers) / len(processed_df) * 100:.2f}%")
                
                # 外れ値処理方法の選択
                st.subheader("外れ値処理方法の選択")
                
                handle_outliers = {}
                for col in numeric_cols:
                    method = st.selectbox(
                        f"{col} の外れ値処理方法",
                        ["処理しない", "クリッピング", "削除"],
                        format_func=lambda x: {
                            "処理しない": "処理しない",
                            "クリッピング": "クリッピング（しきい値に置換）",
                            "削除": "外れ値を含む行を削除"
                        }[x]
                    )
                    
                    if method != "処理しない":
                        handle_outliers[col] = "clip" if method == "クリッピング" else "remove"
                
                # 処理の適用
                if handle_outliers and st.button("外れ値処理を適用"):
                    # 前処理設定の保存
                    st.session_state.preprocessing_config['handle_outliers'] = handle_outliers
                    
                    # データ前処理の実行
                    processed_df = preprocess_data(processed_df, {'handle_outliers': handle_outliers})
                    
                    # 処理後のデータを保存
                    st.session_state.processed_data = processed_df
                    
                    st.success("外れ値処理を適用しました。")
            else:
                st.info("データセットに数値列がありません。")
        
        # スケーリングタブ
        with preprocess_tabs[3]:
            st.subheader("データのスケーリング")
            
            # 数値列のみ処理
            numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_cols:
                # スケーリング方法の選択
                scaling_method = st.selectbox(
                    "スケーリング方法",
                    ["なし", "標準化 (StandardScaler)", "正規化 (MinMaxScaler)", "ロバストスケーラー (RobustScaler)"],
                    format_func=lambda x: {
                        "なし": "スケーリングなし",
                        "標準化 (StandardScaler)": "標準化 (StandardScaler) - 平均0、標準偏差1",
                        "正規化 (MinMaxScaler)": "正規化 (MinMaxScaler) - 0〜1の範囲",
                        "ロバストスケーラー (RobustScaler)": "ロバストスケーラー (RobustScaler) - 外れ値に強い"
                    }[x]
                )
                
                if scaling_method != "なし":
                    # スケーリングする列の選択
                    scaling_columns = st.multiselect(
                        "スケーリングする列を選択",
                        numeric_cols,
                        default=numeric_cols
                    )
                    
                    # 処理の適用
                    if scaling_columns and st.button("スケーリングを適用"):
                        # 前処理設定の保存
                        method_map = {
                            "標準化 (StandardScaler)": "standard",
                            "正規化 (MinMaxScaler)": "minmax",
                            "ロバストスケーラー (RobustScaler)": "robust"
                        }
                        
                        scaling_config = {
                            'scaling': {
                                'method': method_map[scaling_method],
                                'columns': scaling_columns
                            }
                        }
                        
                        st.session_state.preprocessing_config['scaling'] = scaling_config['scaling']
                        
                        # データ前処理の実行
                        processed_df = preprocess_data(processed_df, scaling_config)
                        
                        # 処理後のデータを保存
                        st.session_state.processed_data = processed_df
                        
                        st.success("スケーリングを適用しました。")
            else:
                st.info("データセットに数値列がありません。")
        
        # 特徴量エンジニアリングタブ
        with preprocess_tabs[4]:
            st.subheader("特徴量エンジニアリング")
            
            # 日付列の処理
            datetime_cols = []
            for col in processed_df.columns:
                try:
                    # 日付型の列または日付に変換可能な列を検出
                    if pd.api.types.is_datetime64_any_dtype(processed_df[col]) or pd.to_datetime(processed_df[col], errors='coerce').notna().all():
                        datetime_cols.append(col)
                except:
                    pass
            
            if datetime_cols:
                st.subheader("日付特徴量の抽出")
                
                feature_engineering = {}
                
                for col in datetime_cols:
                    st.write(f"**{col}** から抽出する特徴量:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        extract_year = st.checkbox(f"{col} - 年", value=True)
                        extract_month = st.checkbox(f"{col} - 月", value=True)
                        extract_day = st.checkbox(f"{col} - 日", value=True)
                    
                    with col2:
                        extract_weekday = st.checkbox(f"{col} - 曜日", value=True)
                        extract_quarter = st.checkbox(f"{col} - 四半期", value=False)
                        extract_is_weekend = st.checkbox(f"{col} - 週末フラグ", value=True)
                    
                    # 抽出する特徴量を設定
                    datetime_features = []
                    if extract_year:
                        datetime_features.append('year')
                    if extract_month:
                        datetime_features.append('month')
                    if extract_day:
                        datetime_features.append('day')
                    if extract_weekday:
                        datetime_features.append('weekday')
                    if extract_quarter:
                        datetime_features.append('quarter')
                    if extract_is_weekend:
                        datetime_features.append('is_weekend')
                    
                    if datetime_features:
                        feature_engineering[col] = {'datetime_features': datetime_features}
                
                # 処理の適用
                if feature_engineering and st.button("日付特徴量を抽出"):
                    # 前処理設定の保存
                    st.session_state.preprocessing_config['feature_engineering'] = feature_engineering
                    
                    # データ前処理の実行
                    processed_df = preprocess_data(processed_df, {'feature_engineering': feature_engineering})
                    
                    # 処理後のデータを保存
                    st.session_state.processed_data = processed_df
                    
                    st.success("日付特徴量を抽出しました。")
            else:
                st.info("日付型の列が見つかりませんでした。")
            
            # 数値列のビン分割
            numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_cols:
                st.subheader("数値のビン分割")
                
                # ビン分割する列の選択
                binning_column = st.selectbox(
                    "ビン分割する列を選択",
                    ["なし"] + numeric_cols
                )
                
                if binning_column != "なし":
                    # ビンの数を設定
                    n_bins = st.slider("ビン（区間）の数", 2, 20, 5)
                    
                    # 処理の適用
                    if st.button("ビン分割を適用"):
                        # 特徴量エンジニアリング設定が未初期化なら初期化
                        if 'feature_engineering' not in st.session_state.preprocessing_config:
                            st.session_state.preprocessing_config['feature_engineering'] = {}
                        
                        # ビン分割の設定を追加
                        st.session_state.preprocessing_config['feature_engineering'][binning_column] = {
                            'binning': {
                                'n_bins': n_bins,
                                'labels': None
                            }
                        }
                        
                        # データ前処理の実行
                        processed_df = preprocess_data(processed_df, {
                            'feature_engineering': {
                                binning_column: {
                                    'binning': {
                                        'n_bins': n_bins,
                                        'labels': None
                                    }
                                }
                            }
                        })
                        
                        # 処理後のデータを保存
                        st.session_state.processed_data = processed_df
                        
                        st.success("ビン分割を適用しました。")
        
        # 処理結果タブ
        with preprocess_tabs[5]:
            st.subheader("データ前処理の結果")
            
            # 元のデータと前処理後のデータを比較
            st.write("**元のデータと前処理後のデータの比較:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**元のデータ:**")
                st.dataframe(original_df.head(5))
                st.write(f"行数: {original_df.shape[0]}, 列数: {original_df.shape[1]}")
            
            with col2:
                st.write("**前処理後のデータ:**")
                st.dataframe(processed_df.head(5))
                st.write(f"行数: {processed_df.shape[0]}, 列数: {processed_df.shape[1]}")
            
            # 前処理設定の表示
            st.subheader("適用された前処理:")
            
            if st.session_state.preprocessing_config:
                for process, config in st.session_state.preprocessing_config.items():
                    st.write(f"**{process}:**")
                    st.json(config)
            else:
                st.info("まだ前処理が適用されていません。")
            
            # 前処理のリセット
            if st.button("前処理をリセット"):
                st.session_state.processed_data = st.session_state.data.copy()
                st.session_state.preprocessing_config = {}
                st.success("前処理をリセットしました。元のデータに戻りました。")

# --------------------------------
# 高度な分析機能
# --------------------------------
elif app_mode == "高度な分析":
    st.header("高度な分析")
    
    if st.session_state.data is None:
        st.warning("まずデータを読み込んでください。左のサイドバーから「データ読み込み」を選択してデータを読み込みます。")
    else:
        # データフレームの取得
        df = st.session_state.processed_data
        
        # 分析タイプの選択
        analysis_type = st.sidebar.selectbox(
            "分析タイプを選択",
            ["時系列分析", "クラスター分析", "分布分析"]
        )
        
        # 時系列分析
        if analysis_type == "時系列分析":
            st.subheader("時系列分析")
            
            # 日付列の検出
            datetime_cols = []
            for col in df.columns:
                try:
                    # 日付型の列または日付に変換可能な列を検出
                    if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().all():
                        datetime_cols.append(col)
                except:
                    pass
            
            if not datetime_cols:
                st.warning("日付列が見つかりません。時系列分析を行うには、日付列が必要です。")
            else:
                # 日付列の選択
                date_column = st.selectbox(
                    "日付列を選択",
                    datetime_cols
                )
                
                # 分析対象の数値列を選択
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                
                if not numeric_cols:
                    st.warning("数値列が見つかりません。時系列分析を行うには、数値列が必要です。")
                else:
                    value_column = st.selectbox(
                        "分析対象の数値列を選択",
                        numeric_cols
                    )
                    
                    # 時系列分析の実行
                    if st.button("時系列分析を実行"):
                        with st.spinner("分析を実行中..."):
                            # 日付列が日付型でない場合は変換
                            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                                df[date_column] = pd.to_datetime(df[date_column])
                            
                            # 時系列分析の実行
                            ts_results = perform_time_series_analysis(df, date_column, value_column)
                            
                            # 結果を保存
                            st.session_state.analysis_results['time_series'] = ts_results
                            
                            st.success("分析が完了しました。")
                    
                    # 分析結果の表示
                    if 'time_series' in st.session_state.analysis_results:
                        ts_results = st.session_state.analysis_results['time_series']
                        
                        # 時系列プロットの表示
                        st.subheader("時系列データと移動平均")
                        st.plotly_chart(ts_results['time_series_plot'], use_container_width=True)
                        
                        # 季節分解の表示（ある場合）
                        if 'decomposition_plot' in ts_results:
                            st.subheader("時系列分解（トレンド、季節性、残差）")
                            st.plotly_chart(ts_results['decomposition_plot'], use_container_width=True)
                        
                        # 自己相関・偏自己相関の表示（ある場合）
                        if 'acf_plot' in ts_results and 'pacf_plot' in ts_results:
                            st.subheader("自己相関分析")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(ts_results['acf_plot'], caption="自己相関関数 (ACF)")
                            with col2:
                                st.image(ts_results['pacf_plot'], caption="偏自己相関関数 (PACF)")
                        
                        # エラーの表示（ある場合）
                        if 'error' in ts_results:
                            st.error(ts_results['error'])
        
        # クラスター分析
        elif analysis_type == "クラスター分析":
            st.subheader("クラスター分析")
            
            # 数値列の選択
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("クラスター分析を行うには、少なくとも2つの数値列が必要です。")
            else:
                # 分析に使用する列の選択
                selected_columns = st.multiselect(
                    "分析に使用する列を選択",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if len(selected_columns) < 2:
                    st.warning("少なくとも2つの列を選択してください。")
                else:
                    # クラスター数の設定
                    n_clusters = st.slider("クラスター数", 2, 10, 3)
                    
                    # クラスター分析の実行
                    if st.button("クラスター分析を実行"):
                        with st.spinner("分析を実行中..."):
                            # クラスター分析の実行
                            cluster_results = perform_cluster_analysis(df, selected_columns, n_clusters)
                            
                            # 結果を保存
                            st.session_state.analysis_results['cluster'] = cluster_results
                            
                            st.success("分析が完了しました。")
                    
                    # 分析結果の表示
                    if 'cluster' in st.session_state.analysis_results:
                        cluster_results = st.session_state.analysis_results['cluster']
                        
                        # PCAプロットの表示（ある場合）
                        if 'pca_plot' in cluster_results:
                            st.subheader("PCAによる2次元可視化")
                            st.plotly_chart(cluster_results['pca_plot'], use_container_width=True)
                            
                            # 主成分の寄与度表示（ある場合）
                            if 'loadings_plot' in cluster_results:
                                st.subheader("各変数の主成分への寄与度")
                                st.plotly_chart(cluster_results['loadings_plot'], use_container_width=True)
                        
                        # クラスター統計の表示
                        if 'cluster_stats' in cluster_results:
                            st.subheader("クラスターごとの統計量")
                            st.dataframe(cluster_results['cluster_stats'])
                        
                        # クラスターごとの箱ひげ図表示
                        if 'boxplots' in cluster_results:
                            st.subheader("クラスターごとの特徴分布")
                            for fig in cluster_results['boxplots']:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # クラスタリング結果のダウンロード
                        if 'df_with_clusters' in cluster_results:
                            st.subheader("クラスタリング結果のダウンロード")
                            df_with_clusters = cluster_results['df_with_clusters']
                            
                            # CSVとしてダウンロード
                            csv = df_with_clusters.to_csv(index=False)
                            st.download_button(
                                label="クラスタリング結果をCSVでダウンロード",
                                data=csv,
                                file_name="clustering_results.csv",
                                mime="text/csv"
                            )
        
        # 分布分析
        elif analysis_type == "分布分析":
            st.subheader("変数の分布分析")
            
            # 分析対象の列を選択
            column = st.selectbox(
                "分析する列を選択",
                df.columns.tolist()
            )
            
            if column:
                # 分布分析の実行
                if st.button("分布分析を実行"):
                    with st.spinner("分析を実行中..."):
                        # 分布分析の実行
                        dist_results = perform_distribution_analysis(df, column)
                        
                        # 結果を保存
                        st.session_state.analysis_results['distribution'] = dist_results
                        
                        st.success("分析が完了しました。")
                
                # 分析結果の表示
                if 'distribution' in st.session_state.analysis_results:
                    dist_results = st.session_state.analysis_results['distribution']
                    
                    # 数値列の場合
                    if 'histogram' in dist_results:
                        # ヒストグラムの表示
                        st.subheader("分布ヒストグラム")
                        st.plotly_chart(dist_results['histogram'], use_container_width=True)
                        
                        # 基本統計量の表示
                        if 'stats' in dist_results:
                            st.subheader("基本統計量")
                            st.dataframe(dist_results['stats'])
                        
                        # 歪度と尖度の表示
                        if 'skewness' in dist_results and 'kurtosis' in dist_results:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("歪度 (Skewness)", f"{dist_results['skewness']:.4f}")
                                st.write("0に近いほど対称的な分布。正なら右に裾が長く、負なら左に裾が長い。")
                            with col2:
                                st.metric("尖度 (Kurtosis)", f"{dist_results['kurtosis']:.4f}")
                                st.write("0に近いほど正規分布に近い。正なら尖った分布、負なら平らな分布。")
                        
                        # 正規性検定結果の表示
                        st.subheader("正規性検定")
                        if 'shapiro_test' in dist_results:
                            p_value = dist_results['shapiro_test']['p-value']
                            st.write(f"Shapiro-Wilk検定: p値 = {p_value:.4f}")
                            if p_value < 0.05:
                                st.write("🔴 p < 0.05: 正規分布ではない可能性が高い")
                            else:
                                st.write("🟢 p >= 0.05: 正規分布の可能性がある")
                        
                        if 'normaltest' in dist_results:
                            p_value = dist_results['normaltest']['p-value']
                            st.write(f"D'Agostino's K^2検定: p値 = {p_value:.4f}")
                            if p_value < 0.05:
                                st.write("🔴 p < 0.05: 正規分布ではない可能性が高い")
                            else:
                                st.write("🟢 p >= 0.05: 正規分布の可能性がある")
                        
                        # QQプロットの表示
                        if 'qq_plot' in dist_results:
                            st.subheader("QQプロット (正規性の視覚的確認)")
                            st.image(dist_results['qq_plot'])
                            st.write("直線上に点が並んでいれば正規分布に近い。")
                    
                    # カテゴリ列の場合
                    elif 'bar_chart' in dist_results:
                        # 棒グラフの表示
                        st.subheader("値の分布")
                        st.plotly_chart(dist_results['bar_chart'], use_container_width=True)
                        
                        # 頻度の表示
                        if 'value_counts' in dist_results:
                            st.subheader("出現頻度")
                            st.dataframe(pd.DataFrame({
                                '値': dist_results['value_counts'].index,
                                '頻度': dist_results['value_counts'].values,
                                '割合 (%)': (dist_results['value_counts'].values / dist_results['value_counts'].sum() * 100).round(2)
                            }))

# --------------------------------
# レポート生成機能
# --------------------------------
elif app_mode == "レポート生成":
    st.header("レポート生成")
    
    if st.session_state.data is None:
        st.warning("まずデータを読み込んでください。左のサイドバーから「データ読み込み」を選択してデータを読み込みます。")
    else:
        # データフレームの取得
        df = st.session_state.processed_data
        original_df = st.session_state.data
        
        # レポートの設定
        st.subheader("レポート設定")
        
        report_title = st.text_input("レポートタイトル", "データ分析レポート")
        
        col1, col2 = st.columns(2)
        with col1:
            include_data_preview = st.checkbox("データプレビューを含める", value=True)
            include_basic_stats = st.checkbox("基本統計量を含める", value=True)
            include_data_viz = st.checkbox("データ可視化を含める", value=True)
        
        with col2:
            include_preprocessing = st.checkbox("前処理情報を含める", value=True)
            include_advanced_analysis = st.checkbox("高度な分析結果を含める", value=True)
            include_conclusion = st.checkbox("自動生成された結論を含める", value=True)
        
        # レポート生成ボタン
        if st.button("レポート生成"):
            st.subheader("生成されたレポート")
            
            # レポートの内容を構築
            report_content = f"# {report_title}\n\n"
            report_content += f"**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # データ概要
            report_content += "## 1. データ概要\n\n"
            report_content += f"* 行数: {df.shape[0]}\n"
            report_content += f"* 列数: {df.shape[1]}\n"
            report_content += f"* メモリ使用量: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n"
            report_content += f"* 欠損値の数: {df.isnull().sum().sum()}\n\n"
            
            # データ型情報
            report_content += "### 1.1 データ型情報\n\n"
            dtype_info = pd.DataFrame({
                'データ型': df.dtypes,
                '非欠損値数': df.count(),
                '欠損値数': df.isnull().sum(),
                '欠損率(%)': (df.isnull().sum() / len(df) * 100).round(2),
                'ユニーク値数': df.nunique()
            })
            report_content += dtype_info.to_markdown() + "\n\n"
            
            # データプレビュー
            if include_data_preview:
                report_content += "### 1.2 データプレビュー\n\n"
                report_content += df.head().to_markdown() + "\n\n"
            
            # 基本統計量
            if include_basic_stats:
                report_content += "## 2. 基本統計量\n\n"
                
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if numeric_cols:
                    report_content += "### 2.1 数値列の統計量\n\n"
                    report_content += df[numeric_cols].describe().T.to_markdown() + "\n\n"
                
                non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
                if non_numeric_cols:
                    report_content += "### 2.2 非数値列の情報\n\n"
                    for col in non_numeric_cols:
                        report_content += f"**{col}** のトップ値:\n\n"
                        try:
                            report_content += df[col].value_counts().head(5).to_markdown() + "\n\n"
                        except:
                            report_content += "この列の集計に失敗しました。\n\n"
            
            # 前処理情報
            if include_preprocessing and st.session_state.preprocessing_config:
                report_content += "## 3. 適用された前処理\n\n"
                
                for process, config in st.session_state.preprocessing_config.items():
                    report_content += f"### 3.{list(st.session_state.preprocessing_config.keys()).index(process) + 1} {process}\n\n"
                    report_content += f"```json\n{str(config)}\n```\n\n"
                
                if original_df.shape != df.shape:
                    report_content += "### 3.99 前処理の影響\n\n"
                    report_content += f"* 元のデータ: {original_df.shape[0]} 行 × {original_df.shape[1]} 列\n"
                    report_content += f"* 処理後のデータ: {df.shape[0]} 行 × {df.shape[1]} 列\n"
                    report_content += f"* 変化: {df.shape[0] - original_df.shape[0]} 行, {df.shape[1] - original_df.shape[1]} 列\n\n"
            
            # 高度な分析結果
            if include_advanced_analysis and st.session_state.analysis_results:
                report_content += "## 4. 高度な分析結果\n\n"
                
                # 時系列分析結果
                if 'time_series' in st.session_state.analysis_results:
                    report_content += "### 4.1 時系列分析\n\n"
                    report_content += "時系列分析では、データの時間的パターン、トレンド、季節性、周期性などを調査しました。\n\n"
                    
                    if 'error' in st.session_state.analysis_results['time_series']:
                        report_content += f"**注意:** 分析中にエラーが発生しました: {st.session_state.analysis_results['time_series']['error']}\n\n"
                    else:
                        report_content += "分析の詳細はダッシュボードの「高度な分析」タブで確認できます。\n\n"
                
                # クラスター分析結果
                if 'cluster' in st.session_state.analysis_results:
                    report_content += "### 4.2 クラスター分析\n\n"
                    report_content += "クラスター分析では、データポイントを類似性に基づいてグループ化しました。\n\n"
                    
                    cluster_results = st.session_state.analysis_results['cluster']
                    
                    if 'cluster_stats' in cluster_results:
                        report_content += "**クラスターごとの統計量:**\n\n"
                        report_content += "クラスターごとの基本統計量は、ダッシュボードの「高度な分析」タブで確認できます。\n\n"
                    
                    if 'explained_variance' in cluster_results:
                        report_content += "**主成分分析 (PCA):**\n\n"
                        report_content += f"第1主成分の説明率: {cluster_results['explained_variance'][0]:.2f}\n"
                        report_content += f"第2主成分の説明率: {cluster_results['explained_variance'][1]:.2f}\n"
                        report_content += f"合計説明率: {sum(cluster_results['explained_variance'][:2]):.2f}\n\n"
                
                # 分布分析結果
                if 'distribution' in st.session_state.analysis_results:
                    report_content += "### 4.3 分布分析\n\n"
                    dist_results = st.session_state.analysis_results['distribution']
                    
                    if 'stats' in dist_results:
                        report_content += "**数値変数の分布分析:**\n\n"
                        report_content += dist_results['stats'].to_markdown() + "\n\n"
                        
                        if 'skewness' in dist_results and 'kurtosis' in dist_results:
                            report_content += f"* 歪度 (Skewness): {dist_results['skewness']:.4f}\n"
                            report_content += f"* 尖度 (Kurtosis): {dist_results['kurtosis']:.4f}\n\n"
                        
                        if 'shapiro_test' in dist_results or 'normaltest' in dist_results:
                            report_content += "**正規性検定:**\n\n"
                            
                            if 'shapiro_test' in dist_results:
                                p_value = dist_results['shapiro_test']['p-value']
                                report_content += f"* Shapiro-Wilk検定: p値 = {p_value:.4f}"
                                report_content += f" ({p_value < 0.05 and '正規分布ではない可能性が高い' or '正規分布の可能性がある'})\n"
                            
                            if 'normaltest' in dist_results:
                                p_value = dist_results['normaltest']['p-value']
                                report_content += f"* D'Agostino's K^2検定: p値 = {p_value:.4f}"
                                report_content += f" ({p_value < 0.05 and '正規分布ではない可能性が高い' or '正規分布の可能性がある'})\n\n"
                    
                    elif 'value_counts' in dist_results:
                        report_content += "**カテゴリ変数の分布分析:**\n\n"
                        value_counts = dist_results['value_counts']
                        value_counts_df = pd.DataFrame({
                            '値': value_counts.index,
                            '頻度': value_counts.values,
                            '割合 (%)': (value_counts.values / value_counts.sum() * 100).round(2)
                        })
                        report_content += value_counts_df.to_markdown() + "\n\n"
            
            # 自動生成された結論
            if include_conclusion:
                report_content += "## 5. 結論と洞察\n\n"
                
                # データの基本情報に基づく結論
                report_content += "### 5.1 データの概要\n\n"
                
                # 欠損値に関する結論
                missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                if missing_ratio > 20:
                    report_content += f"* データセットには欠損値が多く（全体の約{missing_ratio:.1f}%）、分析結果の信頼性に影響する可能性があります。\n"
                elif missing_ratio > 0:
                    report_content += f"* データセットには一部欠損値（全体の約{missing_ratio:.1f}%）が存在しますが、適切に処理されています。\n"
                else:
                    report_content += "* データセットに欠損値はなく、完全なデータで分析が行われています。\n"
                
                # 高度な分析に基づく結論
                if st.session_state.analysis_results:
                    report_content += "### 5.2 分析結果からの洞察\n\n"
                    
                    # 時系列分析からの洞察
                    if 'time_series' in st.session_state.analysis_results:
                        report_content += "**時系列分析:**\n\n"
                        report_content += "* 時系列データの詳細なパターンやトレンドは、ダッシュボードのグラフで視覚的に確認できます。\n"
                        if 'decomposition_plot' in st.session_state.analysis_results['time_series']:
                            report_content += "* データは季節性の要素とトレンド成分に分解され、時間的パターンの理解が深まりました。\n"
                    
                    # クラスター分析からの洞察
                    if 'cluster' in st.session_state.analysis_results:
                        report_content += "**クラスター分析:**\n\n"
                        cluster_results = st.session_state.analysis_results['cluster']
                        if 'df_with_clusters' in cluster_results:
                            n_clusters = len(cluster_results['df_with_clusters']['cluster'].unique())
                            report_content += f"* データは{n_clusters}つの異なるクラスターに分類され、それぞれ特徴的なパターンが示されています。\n"
                        if 'explained_variance' in cluster_results:
                            total_var = sum(cluster_results['explained_variance'][:2])
                            if total_var > 0.7:
                                report_content += f"* 2つの主成分で元の変動の{total_var:.0%}を説明でき、データの次元削減に成功しています。\n"
                            else:
                                report_content += f"* 2つの主成分では元の変動の{total_var:.0%}しか説明できず、データの複雑性が示唆されています。\n"
                    
                    # 分布分析からの洞察
                    if 'distribution' in st.session_state.analysis_results:
                        report_content += "**分布分析:**\n\n"
                        dist_results = st.session_state.analysis_results['distribution']
                        if 'skewness' in dist_results:
                            skew = dist_results['skewness']
                            if abs(skew) < 0.5:
                                report_content += f"* 分析した変数はほぼ対称的な分布を示しています（歪度: {skew:.2f}）。\n"
                            elif skew > 0:
                                report_content += f"* 分析した変数は右に裾が長い分布を示しています（歪度: {skew:.2f}）。\n"
                            else:
                                report_content += f"* 分析した変数は左に裾が長い分布を示しています（歪度: {skew:.2f}）。\n"
                        
                        if 'normaltest' in dist_results:
                            p_value = dist_results['normaltest']['p-value']
                            if p_value < 0.05:
                                report_content += "* 正規性検定の結果、データは正規分布に従っていない可能性が高いです。\n"
                            else:
                                report_content += "* 正規性検定の結果、データは正規分布に従っている可能性があります。\n"
                
                # 総括
                report_content += "### 5.3 総括\n\n"
                report_content += "このレポートでは、データの基本的な特性、前処理の影響、そして高度な分析結果を提示しました。\n"
                report_content += "より詳細な分析や視覚化はダッシュボードで利用できます。\n\n"
                report_content += "データから得られた主な洞察は以下の通りです：\n\n"
                
                if 'time_series' in st.session_state.analysis_results:
                    report_content += "* 時系列データは時間的パターンを示しており、予測モデルの構築に役立つ可能性があります。\n"
                
                if 'cluster' in st.session_state.analysis_results:
                    report_content += "* データポイントはいくつかの明確なクラスターに分類でき、それぞれ特徴的な属性を持っています。\n"
                
                if 'distribution' in st.session_state.analysis_results:
                    report_content += "* 変数の分布特性を理解することで、異常値の検出や適切な統計モデルの選択が可能になります。\n"
                
                report_content += "\n**注意**: このレポートは自動生成されたものです。詳細な解釈には専門家の判断が必要な場合があります。"
            
            # レポートの表示
            st.markdown(report_content)
            
            # ダウンロードボタン
            st.download_button(
                label="レポートをMarkdownでダウンロード",
                data=report_content,
                file_name=f"{report_title.lower().replace(' ', '_')}.md",
                mime="text/markdown"
            )

# フッター
st.markdown("---")
st.markdown("© 2023 データ分析ダッシュボード - Pythonで作成")