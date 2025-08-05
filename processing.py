
# 辞書読み込み
def return_pickle(file_path):
    with open(file_path, "rb") as jd:
        mapping = pickle.load(jd)
    
    return mapping

# floatに変換
def convert_to_float_if_possible(df):
    df_converted = df.copy()
    for col in df.columns:
        try:
            # 変換を試みる（NaNは発生させたくないので errors='raise'）
            converted = pd.to_numeric(df[col], errors='raise')
            # print(col, converted.dtype)
            if pd.api.types.is_numeric_dtype(converted):
                df_converted[col] = converted
        except:
            pass  # 変換できなかった列は無視
    return df_converted

# データの初期加工
def df_first_processing(df):
    df = df.copy()

    # 血統pickle作成
    femal_mapping = return_pickle(femal_horse_path)
    horse_mapping = return_pickle(horse_path)

    # マッピング
    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_mapping), errors='coerce')

    # 特殊記号を消す
    df['騎手'] = df['騎手'].str.replace('▲', '')
    df['騎手'] = df['騎手'].str.replace('△', '')
    df['騎手'] = df['騎手'].str.replace('☆', '')
    df['騎手'] = df['騎手'].str.replace('★', '')
    df['騎手'] = df['騎手'].str.replace('◇', '')

    # 1つのカラムに入っているデータを複数カラムに分ける
    df_sex = df['性齢'].str.extract(r'([牝牡セ])(\d+)', expand=True)
    df['性'] = df_sex.loc[:, 0]
    df['齢'] = df_sex.loc[:, 1]

    df_weight = df['馬体重(増減)'].str.extract(r'(\d{3}).([+-0]\d*)', expand=True)
    df['馬体重'] = df_weight.loc[:, 0]
    df['体重増減'] = df_weight.loc[:, 1].str.replace('\+', '', regex=True)

    # 性別をマッピング
    sex_mapping = {'牡': 1, '牝': 2, 'セ': 3}
    df['性'] = df['性'].map(sex_mapping)

    df['フィールド'] = df['フィールド'].map(field_mapping)
    df['馬場'] = df['馬場'].map(condition_mapping)

    # 騎手のユニーク値から辞書をつくる
    jockey_mapping = return_pickle(jockey_path)
    jwin_mapping = return_pickle(jwin_path)

    df['騎手'] = df['騎手'].map(jockey_mapping)

    df['騎手'] = pd.to_numeric(df['騎手'].map(jwin_mapping), errors='coerce')

    # 「性齢」「馬体重（増減）」はいらないので消す
    df = df.drop(['性齢', '馬体重(増減)', '馬体重', '体重増減'], axis=1)

    df = convert_to_float_if_possible(df)

    return df

# 前走~5走前のデータを処理
def df_big_past_processing(df):
    df_all = df.copy()
    # 騎手の辞書を読み込み
    jockey_mapping = return_pickle(jockey_path)

    # 以降2~5走の処理
    for sou in range(1, 6):
        sou = str(sou)
        if int(sou) == 1:
            df_split = df['前走'].str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d)(.+)(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
        else:
            df_split = df[sou+'走'].str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d)(.+)(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
        df_split.columns = ['日付', sou+'場所', sou+'過去着順', sou+'レース名', sou+'フィールド', sou+'距離', sou+'タイム', sou+'馬場', sou+'出走馬数', sou+'馬番', sou+'人気', sou+'騎手', sou+'斤量', sou+'コーナー通過順', sou+'後3F', sou+'馬体重', sou+'体重増減', sou+'着差']

        # 2~5走のカラムを削除
        df_split = df_split.drop(['日付'], axis=1)

        # 4角コーナー通過順のみに
        df_split[sou+'コーナー通過順'] = df_split[sou+'コーナー通過順'].str[-4:-1].apply(lambda x : x if x != ' ' else None)
        df_split[sou+'コーナー通過順'] = df_split[sou+'コーナー通過順'].astype(float).abs()

        # クラス別に分類
        df_split[sou+'クラス'] = 0
        for k, v in class_dict.items():
            df_split[sou+'クラス'] = df_split[sou+'クラス'].mask(df_split[sou+'レース名'].str.contains(k, na=False), v)


        # 文字列データを数値データにする
        df_split[sou+'場所'] = df_split[sou+'場所'].map(nagoya_mapping)

        df_split[sou+'フィールド'] = df_split[sou+'フィールド'].map(field_mapping)

        df_split[sou+'馬場'] = df_split[sou+'馬場'].map(condition_mapping)

        # 文字列から数値に変換する
        df_split[sou+'騎手'] = df_split[sou+'騎手'].map(jockey_mapping)

        # タイムを秒表記にする
        df_split[sou+'タイム'] = pd.to_datetime(df_split[sou+'タイム'], format='%M:%S.%f') - base_time
        df_split[sou+'タイム'] = df_split[sou+'タイム'].dt.total_seconds()

        # # スピード指数の計算
        for i in range(1, 11):
            for k in range(1000, 3700, 100):
                try:
                    speed = speed_dict[f'{i}{k}1']
                    df_split.loc[(df_split[sou+'場所'] == i) & (df_split[sou+'距離'].astype(float) == k) & (df_split[sou+'フィールド'] == 1), sou+'スピード指数'] = ((speed + 0.01 - df_split[sou+'タイム'].astype(float)) * 10) * (1 / speed * 100) + (df_split[sou+'馬場'] * 10) + (df_split[sou+'斤量'].astype(float) - 55) * 2 + 80
                except KeyError:
                    pass
                try:
                    speed = speed_dict[f'{i}{k}2']
                    df_split.loc[(df_split[sou+'場所'] == i) & (df_split[sou+'距離'].astype(float) == k) & (df_split[sou+'フィールド'] == 2), sou+'スピード指数'] = ((speed + 0.01 - df_split[sou+'タイム'].astype(float)) * 10) * (1 / speed * 100) + (13 - df_split[sou+'馬場'] * 3) + (df_split[sou+'斤量'].astype(float) - 55) * 2 + 80
                except KeyError:
                    pass

        # クラス別に分類
        df_split[sou+'クラス'] = 0
        for k, v in class_dict.items():
            df_split[sou+'クラス'] = df_split[sou+'クラス'].mask(df_split[sou+'レース名'].str.contains(k, na=False), v)

        # df_split[sou+'クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_split[sou+'クラス'].astype(str) + df_split[sou+'過去着順'].astype(str), errors='coerce')

        df_split[sou+'着差'] = df_split[sou+'着差'].astype(float) + (df_split[sou+'クラス'].astype(int) * 0.5)

        # 上がり3Fを指数化
        df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 1) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (0.94 + (df_split[sou+'距離'].astype(float) / 20000)))
        df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 2) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (1.01 + (df_split[sou+'距離'].astype(float) / 20000)))
        df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 3) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (0.36 + (df_split[sou+'距離'].astype(float) * 1.5 / 100000)))

        # print(df_split[sou+'場所'].dtype, df_split[sou+'距離'].dtype, df_split[sou+'フィールド'].dtype, df['フィールド'].dtype, df['距離'].dtype)
        df_split[sou+'距離差'] = df['距離'].astype(float) - df_split[sou+'距離'].astype(float)
        df_split[sou+'場所変化'] = df_split[sou+'場所'] - field
        df_split[sou+'フィールド変化'] = df_split[sou+'フィールド'] - df['フィールド']

        # 不要なカラムを削除
        # df_split = df_split.drop([sou+'レース名', sou+'騎手',sou+'場所',sou+'フィールド',sou+'馬場',sou+'タイム',sou+'出走馬数', sou+'馬番',sou+'馬体重', sou+'体重増減',sou+'斤量'], axis=1)
        df_split = df_split.drop([sou+'レース名', sou+'騎手'], axis=1)


        # 今走と過去走を結合
        df_all = pd.concat([df_all, df_split], axis=1)

        if int(sou) == 1:
            past_level(df_all)

    return df_all

# 過去走の平均クラスと平均ペースを算出
def past_level(df_all):
    df_all = df_all.copy()
    # クエリListを作成
    id_count = df_all['レースID'].value_counts(sort=False)
    n_list = id_count.values.tolist()

    n_list = n_list[:-1]
    n = 0
    df_all['平均クラス'] = np.nan
    df_all['平均ペース'] = np.nan
    for i in n_list:
        n += i
        df_all.iloc[n-i:n, df_all.columns.get_loc('平均クラス')] = df_all.iloc[n-i:n, df_all.columns.get_loc('1クラス')].mean().astype(int)
        df_all.iloc[n-i:n, df_all.columns.get_loc('平均ペース')] = df_all.iloc[n-i:n, df_all.columns.get_loc('1コーナー通過順')].mean()

    df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
    df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']

    return df_all

# テストデータを分離してターゲットエンコーディング
def encording(df_all):
    df_all = df_all.copy()
    
    sire_mapping = return_pickle(sire_path)

    df_all = df_all.map(sire_mapping)

    return df_all

# 終盤のデータ加工
def df_end_processing(df_all):
    df_all = df_all.copy()

    # 着差とスピード指数のbest, avカラム作成
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    df_all = df_all.drop(['前走', '2走', '3走', '4走', '5走', '母父馬'], axis=1)

    print(df_all.isnull().sum())

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # 上昇度カラム作成
    df_all['5過去着順'] = df_all['5過去着順'].astype(float)
    df_all['4過去着順'] = df_all['4過去着順'].astype(float)
    df_all['3過去着順'] = df_all['3過去着順'].astype(float)
    df_all['2過去着順'] = df_all['2過去着順'].astype(float)
    df_all['1過去着順'] = df_all['1過去着順'].astype(float)
    df_all['上昇度'] = (df_all['5過去着順'] - df_all['4過去着順']) + (df_all['4過去着順'] - df_all['3過去着順']) + (df_all['3過去着順'] - df_all['2過去着順']) + (df_all['2過去着順'] - df_all['1過去着順']) / (df_all[['1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順']].isnull().sum(axis=1) + 1)

    # 型変換
    df_all = convert_to_float_if_possible(df_all)
    
    return df_all

