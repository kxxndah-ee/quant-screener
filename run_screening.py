import os
import io
import datetime
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import urllib3

# Windows SSL 호환 및 경고 무시
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

urllib3.disable_warnings()

from pykrx import stock
import FinanceDataReader as fdr

# ==========================================
# 1. 폰트 및 시각화 기본 설정
# ==========================================
os_name = platform.system()
font_family = 'Malgun Gothic'
if os_name == 'Darwin':       # macOS
    font_family = 'AppleGothic'
elif os_name == 'Windows':    # Windows
    font_family = 'Malgun Gothic'
else:                         # Linux / Colab / Docker
    font_family = 'NanumGothic'

sns.set_style("whitegrid", {
    "font.family": font_family,
    "axes.unicode_minus": False
})
plt.rc('font', family=font_family)
plt.rcParams['font.family'] = font_family
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 2. 데이터 보조 수집기 (KRX 로그인 차단 시 무결성 대체)
# ==========================================
def fetch_krx_via_naver_api():
    """
    KRX data.krx.co.kr의 비로그인 차단(400 LOGOUT) 발생 시,
    네이버 증권 공식 API를 통해 KRX 전 종목의 밸류에이션 및 시가총액 데이터를 수집합니다.
    """
    url = "https://stock.naver.com/api/domestic/market/stock/default?page=1&pageSize=3500"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    raw_list = resp.json()

    records = []
    for item in raw_list:
        ticker = str(item.get("itemcode", "")).strip()
        if not ticker:
            continue
        
        name = item.get("itemname", "")
        close_p = float(item.get("nowPrice", 0) or 0)
        market_cap = float(item.get("marketSum", 0) or 0)
        
        # PBR, PER, EPS, ROE, BPS, DIV
        pbr_raw = item.get("pbr")
        per_raw = item.get("per")
        roe_raw = item.get("roe")
        eps_raw = item.get("eps")
        div_raw = item.get("dividendRate")
        dps_raw = item.get("dividend")

        pbr = float(pbr_raw) if pbr_raw is not None and str(pbr_raw).strip() != '' else np.nan
        per = float(per_raw) if per_raw is not None and str(per_raw).strip() != '' else np.nan
        roe = float(roe_raw) if roe_raw is not None and str(roe_raw).strip() != '' else np.nan
        eps = float(eps_raw) if eps_raw is not None and str(eps_raw).strip() != '' else np.nan
        div = float(div_raw) if div_raw is not None and str(div_raw).strip() != '' else np.nan
        dps = float(dps_raw) if dps_raw is not None and str(dps_raw).strip() != '' else np.nan
        
        bps = np.nan
        if pbr > 0 and close_p > 0:
            bps = close_p / pbr

        records.append({
            "티커": ticker,
            "종목명": name,
            "종가": close_p,
            "시가총액": market_cap,
            "PBR": pbr,
            "PER": per,
            "ROE": roe,
            "EPS": eps,
            "BPS": bps,
            "DIV": div,
            "DPS": dps
        })

    df_all = pd.DataFrame(records)
    
    # pykrx의 df_fundamental 형태 (index: 티커, columns: BPS, PER, PBR, EPS, DIV, DPS)
    df_fundamental = df_all[['티커', 'BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']].set_index('티커')
    # pykrx의 df_market_cap 형태 (index: 티커, columns: 종가, 시가총액 등)
    df_market_cap = df_all[['티커', '종가', '시가총액']].set_index('티커')
    
    return df_fundamental, df_market_cap, df_all


def get_krx_sector_info():
    """
    KRX 상장법인상세목록(KIND)을 통해 업종(Sector) 및 주요제품(Industry) 정보를 수집합니다.
    """
    # 1. 로컬 SQLite DB 확인
    quant_screener_db = r"C:\Users\user\.gemini\antigravity\scratch\quant_screener\data\app.db"
    if os.path.exists(quant_screener_db):
        try:
            import sqlite3
            conn = sqlite3.connect(quant_screener_db)
            df_u = pd.read_sql("SELECT code, sector, industry FROM universe WHERE is_active=1", conn)
            conn.close()
            if not df_u.empty:
                df_u.rename(columns={"code": "Code", "sector": "Sector", "industry": "Industry"}, inplace=True)
                df_u["Code"] = df_u["Code"].astype(str).str.zfill(6)
                return df_u
        except Exception:
            pass

    # 2. KIND 웹 다운로드
    try:
        url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        r = requests.get(url, verify=False, timeout=15)
        r.encoding = "cp949"
        dfs = pd.read_html(io.StringIO(r.text))
        if dfs and not dfs[0].empty:
            df_k = dfs[0]
            col_map = {}
            for c in df_k.columns:
                if "종목코드" in c or "코드" in c:
                    col_map[c] = "Code"
                elif "업종" in c:
                    col_map[c] = "Sector"
                elif "주요제품" in c:
                    col_map[c] = "Industry"
            df_k = df_k.rename(columns=col_map)
            df_k["Code"] = df_k["Code"].astype(str).str.zfill(6)
            return df_k[["Code", "Sector", "Industry"]]
    except Exception as e:
        print(f"KIND 업종 정보 수집 경고: {e}")

    return pd.DataFrame(columns=["Code", "Sector", "Industry"])


# ==========================================
# 3. 데이터 수집 및 전처리 함수
# ==========================================
def fetch_and_process_data(csv_filename="low_pbr_peer_list.csv"):
    today = datetime.datetime.today()
    target_date = today.strftime("%Y%m%d")
    
    # pykrx 시도
    use_fallback = False
    try:
        nearest_date = stock.get_nearest_business_day_in_a_week(today.strftime("%Y%m%d"))
        if nearest_date:
            target_date = nearest_date
        print(f">> [{target_date}] 기준 pykrx 데이터 수집 시도...")
        df_fundamental = stock.get_market_fundamental_by_ticker(target_date, market="ALL")
        df_market_cap = stock.get_market_cap_by_ticker(target_date, market="ALL")
        if df_fundamental.empty or df_market_cap.empty:
            use_fallback = True
    except Exception as e:
        print(f">> pykrx KRX 직접 호출 제한 (data.krx.co.kr 비로그인 세션): {e}")
        use_fallback = True

    if use_fallback:
        print(f">> 최신 실시간 KRX 전 종목 데이터 엔진으로 대체 수집 중...")
        df_fundamental, df_market_cap, _ = fetch_krx_via_naver_api()

    # FinanceDataReader: 업종 분류 매핑
    df_listing = fdr.StockListing('KRX')
    
    # FDR 최신 버전 호환: Sector, Industry 보강
    if 'Sector' not in df_listing.columns or 'Industry' not in df_listing.columns:
        df_sec_info = get_krx_sector_info()
        df_listing = df_listing.merge(df_sec_info, on='Code', how='left')

    df_sector = df_listing[['Code', 'Name', 'Market', 'Sector', 'Industry']].copy()
    df_sector.rename(columns={'Code': '티커', 'Name': '종목명', 'Market': '시장'}, inplace=True)
    df_sector.set_index('티커', inplace=True)

    # 병합
    df = df_fundamental.join(df_market_cap[['종가', '시가총액']], how='inner')
    df = df.join(df_sector, how='left').reset_index()
    df.rename(columns={'index': '티커'}, inplace=True)

    # PBR 1.0 미만 필터링 (0 이하 및 결측치 제외)
    df_filtered = df[(df['PBR'] > 0) & (df['PBR'] < 1.0)].copy()

    # ROE 계산: ROE(%) = (PBR / PER) * 100 또는 EPS / BPS
    if 'ROE' not in df_filtered.columns or df_filtered['ROE'].isna().all():
        if 'EPS' in df_filtered.columns and 'BPS' in df_filtered.columns:
            df_filtered['ROE'] = (df_filtered['EPS'] / df_filtered['BPS']) * 100
        else:
            df_filtered['ROE'] = np.where(df_filtered['PER'] > 0, (df_filtered['PBR'] / df_filtered['PER']) * 100, np.nan)
    else:
        # 이미 ROE 컬럼이 있는 경우 결측치 보완
        cond_calc = df_filtered['ROE'].isna() & (df_filtered['PER'] > 0)
        df_filtered.loc[cond_calc, 'ROE'] = (df_filtered.loc[cond_calc, 'PBR'] / df_filtered.loc[cond_calc, 'PER']) * 100

    # 건설 및 인프라 피어그룹 태깅
    construction_keywords = ['건설', '토목', '엔지니어링', '인프라', '설비']
    def classify_target(row):
        sector_text = f"{row.get('Sector', '')} {row.get('Industry', '')}"
        if str(row['티커']) == '003070':
            return '코오롱글로벌'
        for kw in construction_keywords:
            if kw in sector_text:
                return '건설/인프라 피어'
        return '일반'

    df_filtered['타깃그룹'] = df_filtered.apply(classify_target, axis=1)

    # PBR 기준 오름차순 정렬 및 CSV 저장
    df_sorted = df_filtered.sort_values(by='PBR', ascending=True)
    cols = ['티커', '종목명', '시장', '타깃그룹', 'Sector', 'Industry', '종가', '시가총액', 'PBR', 'PER', 'ROE', 'BPS', 'DIV']
    available_cols = [c for c in cols if c in df_sorted.columns]
    df_final = df_sorted[available_cols]
    
    df_final.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f">> CSV 파일 생성 완료: {os.path.abspath(csv_filename)}")
    print(f">> 총 {len(df_final)}개 저PBR 종목 발굴 (건설/인프라 피어: {(df_final['타깃그룹'] != '일반').sum()}개)")

    return df_final, target_date


# ==========================================
# 4. PBR-ROE 산점도 시각화 함수
# ==========================================
def plot_pbr_roe_matrix(df, target_date, chart_filename="pbr_roe_matrix.png"):
    peer_df = df[df['타깃그룹'].isin(['건설/인프라 피어', '코오롱글로벌'])].dropna(subset=['PBR', 'ROE']).copy()

    # 극단치 제거 (시각화 가독성 확보)
    peer_df = peer_df[(peer_df['PBR'] > 0) & (peer_df['PBR'] <= 1.2)]
    peer_df = peer_df[(peer_df['ROE'] >= -15) & (peer_df['ROE'] <= 25)]

    if peer_df.empty:
        print(">> 유효한 건설/인프라 피어 데이터가 없어 차트를 생성하지 못했습니다.")
        return

    # 스타일 및 한글 폰트 적용
    sns.set_style("whitegrid", {
        "font.family": font_family,
        "axes.unicode_minus": False
    })
    plt.rc('font', family=font_family)
    plt.rcParams['font.family'] = font_family
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=300)

    mean_pbr = peer_df['PBR'].mean()
    mean_roe = peer_df['ROE'].mean()

    # 사분면 분할 기준선 (평균)
    ax.axhline(mean_roe, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(mean_pbr, color='grey', linestyle='--', linewidth=1, alpha=0.7)

    # 1) 일반 피어 플롯
    peers = peer_df[peer_df['타깃그룹'] != '코오롱글로벌']
    ax.scatter(peers['PBR'], peers['ROE'], c='#3498db', s=140, alpha=0.75, edgecolors='#1b4f72', linewidth=0.8, label='건설/인프라 피어')

    # 2) 코오롱글로벌 강조 플롯
    kolon = peer_df[peer_df['타깃그룹'] == '코오롱글로벌']
    if not kolon.empty:
        ax.scatter(kolon['PBR'], kolon['ROE'], c='#e74c3c', s=420, marker='*', edgecolors='black', linewidth=1.5, label='코오롱글로벌 (003070)', zorder=5)

    # 3) 종목명 라벨링
    for _, row in peer_df.iterrows():
        is_target = (row['타깃그룹'] == '코오롱글로벌')
        weight = 'bold' if is_target else 'normal'
        color = '#c0392b' if is_target else '#2c3e50'
        font_size = 11 if is_target else 8.5
        y_offset = 0.45 if is_target else 0.25

        ax.annotate(row['종목명'], (row['PBR'], row['ROE'] + y_offset), fontsize=font_size, fontweight=weight, color=color, ha='center', va='bottom', fontfamily=font_family)

    # 4) 사분면 해석 박스
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.text(x_min + (mean_pbr - x_min)*0.05, y_max - (y_max - mean_roe)*0.08,
            "★ 저평가 고효율 매력 구간\n(상대적 고ROE & 극저PBR)", 
            fontsize=10, fontweight='bold', color='#27ae60', fontfamily=font_family,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1", edgecolor="#2ecc71", alpha=0.85))

    ax.set_title(f"건설·인프라 피어그룹 PBR-ROE 밸류에이션 매트릭스 ({target_date} 기준)", fontsize=15, fontweight='bold', pad=15, fontfamily=font_family)
    ax.set_xlabel("PBR (배) → 우측으로 갈수록 고평가", fontsize=11, fontweight='bold', labelpad=10, fontfamily=font_family)
    ax.set_ylabel("ROE (%) → 상단으로 갈수록 수익성 양호", fontsize=11, fontweight='bold', labelpad=10, fontfamily=font_family)
    ax.legend(loc='lower right', frameon=True, fontsize=10, prop={'family': font_family})

    plt.tight_layout()
    plt.savefig(chart_filename, dpi=300)
    plt.close()
    print(f">> 차트 이미지 생성 완료: {os.path.abspath(chart_filename)}")


# ==========================================
# 5. 실행 진입점
# ==========================================
if __name__ == "__main__":
    df_result, last_date = fetch_and_process_data()
    plot_pbr_roe_matrix(df_result, last_date)
    print(">> 모든 작업이 완료되었습니다.")
