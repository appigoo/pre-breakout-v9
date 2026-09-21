import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from itertools import combinations

st.set_page_config(page_title='爆升前決策引擎 V2.4', layout='wide')

# =========================
# 資料層
# =========================
@st.cache_data(ttl=3600)
def load_data(symbol, period='5y'):
    df = yf.download(symbol, period=period, interval='1d', auto_adjust=False, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

@st.cache_data(ttl=3600)
def load_macro(period='5y'):
    return load_data('QQQ', period), load_data('^VIX', period)


def add_features(df, qqq=None, vix=None):
    x = df.copy()
    c, o, h, l, v = x['Close'], x['Open'], x['High'], x['Low'], x['Volume']
    x['ret_1d'] = c.pct_change()
    x['ret_3d'] = c.pct_change(3)
    x['ret_5d'] = c.pct_change(5)
    x['ret_10d'] = c.pct_change(10)
    x['ret_20d'] = c.pct_change(20)
    x['vol_ratio'] = v / v.rolling(20).mean()
    x['vol_chg_5d'] = v.pct_change(5)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    x['macd'] = ema12 - ema26
    x['macd_signal'] = x['macd'].ewm(span=9, adjust=False).mean()
    x['macd_hist'] = x['macd'] - x['macd_signal']
    x['macd_hist_change'] = x['macd_hist'].diff()
    x['macd_hist_up3'] = x['macd_hist'].diff().gt(0).rolling(3).sum().eq(3)
    x['bull_candle'] = c > o
    x['range_pct'] = (h - l) / c
    x['body_pct'] = (c - o).abs() / c
    x['close_position'] = (c - l) / (h - l).replace(0, np.nan)
    x['breakout_20_level'] = h.rolling(20).max().shift(1)
    x['breakout_20'] = c > x['breakout_20_level']
    x['high_20_ratio'] = c / x['breakout_20_level'] - 1
    x['atr'] = (h - l).rolling(14).mean()
    x['atr_pct'] = x['atr'] / c
    x['swing_low_10'] = l.rolling(10).min().shift(1)
    if qqq is not None and not qqq.empty:
        q = qqq['Close'].reindex(x.index).ffill()
        x['qqq_1d'] = q.pct_change()
        x['qqq_5d'] = q.pct_change(5)
        x['relative_strength_5d'] = x['ret_5d'] - x['qqq_5d']
    if vix is not None and not vix.empty:
        vv = vix['Close'].reindex(x.index).ffill()
        x['vix_level'] = vv
        x['vix_5d'] = vv.pct_change(5)
    return x


def make_labels(df, forward_days, target, cooldown):
    x = df.copy()
    highs = x['High'].to_numpy(float)
    closes = x['Close'].to_numpy(float)
    n = len(x)
    future_max = np.full(n, np.nan)
    future_close = np.full(n, np.nan)
    future_min = np.full(n, np.nan)
    for i in range(n - forward_days):
        path = highs[i+1:i+forward_days+1]
        lows = x['Low'].to_numpy(float)[i+1:i+forward_days+1]
        cls = x['Close'].to_numpy(float)[i+1:i+forward_days+1]
        future_max[i] = np.max(path) / closes[i] - 1
        future_min[i] = np.min(lows) / closes[i] - 1
        future_close[i] = cls[-1] / closes[i] - 1
    event = np.zeros(n, dtype=bool)
    last = -cooldown - 1
    for i in range(n - forward_days):
        if future_max[i] >= target and i - last > cooldown:
            event[i] = True
            last = i
    x['future_max_return'] = future_max
    x['future_min_return'] = future_min
    x['future_close_return'] = future_close
    x['target_hit'] = future_max >= target
    x['explosion_event'] = event
    return x


# =========================
# V2.4 Trend / Volume / Candle Gate
# =========================
def add_v24_features(x):
    x=x.copy(); c,o,h,l,v=x['Close'],x['Open'],x['High'],x['Low'],x['Volume']
    x['ema20']=c.ewm(span=20,adjust=False).mean(); x['ema50']=c.ewm(span=50,adjust=False).mean(); x['ema200']=c.ewm(span=200,adjust=False).mean()
    x['ema20_slope_5d']=x['ema20'].pct_change(5); x['ema50_slope_10d']=x['ema50'].pct_change(10)
    x['price_vs_ema20']=c/x['ema20']-1; x['price_vs_ema50']=c/x['ema50']-1; x['price_vs_ema200']=c/x['ema200']-1
    x['swing_low_20']=l.rolling(20).min().shift(1); x['recent_high_10']=h.rolling(10).max().shift(1)
    x['lower_high']=x['recent_high_10'] < x['recent_high_10'].shift(10)
    x['higher_low']=x['swing_low_10'] > x['swing_low_10'].shift(10)
    prev_o,prev_c=o.shift(1),c.shift(1)
    x['bearish_engulfing']=(c<o)&(prev_c>prev_o)&(o>=prev_c)&(c<=prev_o)
    x['bullish_engulfing']=(c>o)&(prev_c<prev_o)&(o<=prev_c)&(c>=prev_o)
    x['large_bear_high_vol']=(x['ret_1d']<-0.025)&(x['vol_ratio']>=1.5)
    x['large_bull_high_vol']=(x['ret_1d']>0.025)&(x['vol_ratio']>=1.5)
    return x

def trend_components(r):
    return {
        '價格>EMA20': bool(pd.notna(r.get('price_vs_ema20')) and r['price_vs_ema20']>0),
        'EMA20>EMA50': bool(pd.notna(r.get('ema20')) and pd.notna(r.get('ema50')) and r['ema20']>r['ema50']),
        'EMA50>EMA200': bool(pd.notna(r.get('ema50')) and pd.notna(r.get('ema200')) and r['ema50']>r['ema200']),
        'EMA20向上': bool(pd.notna(r.get('ema20_slope_5d')) and r['ema20_slope_5d']>0),
        'EMA50向上': bool(pd.notna(r.get('ema50_slope_10d')) and r['ema50_slope_10d']>0),
        'Higher Low': bool(r.get('higher_low',False)),
        '非Lower High': not bool(r.get('lower_high',False)),
        '相對QQQ強勢': bool(pd.notna(r.get('relative_strength_5d')) and r['relative_strength_5d']>0),
    }

def trend_score(r):
    d=trend_components(r); return 100*sum(d.values())/len(d),d

def volume_pressure(r):
    vr,ret=r.get('vol_ratio',np.nan),r.get('ret_1d',np.nan)
    if pd.isna(vr) or pd.isna(ret): return '⚪ UNKNOWN',50
    if vr>=1.5 and ret<-0.02: return '🔴 放量下跌／賣壓',15
    if vr>=1.5 and ret>0.02: return '🟢 放量上升／買盤',90
    if vr>=1.3 and ret<0: return '🟠 偏賣壓',35
    if vr>=1.3 and ret>0: return '🟢 偏買盤',75
    return '⚪ 一般',50

def candle_score(r):
    score=50; labels=[]
    if bool(r.get('large_bear_high_vol',False)): score-=25; labels.append('放量大陰燭')
    if bool(r.get('bearish_engulfing',False)): score-=20; labels.append('Bearish Engulfing')
    if bool(r.get('large_bull_high_vol',False)): score+=25; labels.append('放量大陽燭')
    if bool(r.get('bullish_engulfing',False)): score+=20; labels.append('Bullish Engulfing')
    cp=r.get('close_position',np.nan)
    if pd.notna(cp):
        if cp<0.30: score-=10; labels.append('收市接近低位')
        elif cp>0.70: score+=10; labels.append('收市接近高位')
    return float(np.clip(score,0,100)),labels

FEATURE_RULES = {
    '5日升幅≥3%': lambda x: x['ret_5d'] >= 0.03,
    '成交量≥1.5倍': lambda x: x['vol_ratio'] >= 1.5,
    '成交量≥2倍': lambda x: x['vol_ratio'] >= 2.0,
    'MACD柱連升3日': lambda x: x['macd_hist_up3'],
    '陽燭': lambda x: x['bull_candle'],
    '20日突破': lambda x: x['breakout_20'],
    'QQQ 5日≥1%': lambda x: x['qqq_5d'] >= 0.01,
    'VIX 5日下跌': lambda x: x['vix_5d'] < 0,
    '相對QQQ強勢': lambda x: x['relative_strength_5d'] > 0,
    '20日高點附近': lambda x: x['high_20_ratio'] >= -0.02,
    '收市接近當日高位': lambda x: x['close_position'] >= 0.70,
    '波幅擴張': lambda x: x['atr_pct'] >= 0.025,
}


def rule_matrix(df):
    out = pd.DataFrame(index=df.index)
    for name, fn in FEATURE_RULES.items():
        try:
            out[name] = fn(df).fillna(False).astype(bool)
        except Exception:
            out[name] = False
    return out


def search_combinations(train_df, target, min_signals=15, max_features=3, top_n=20):
    valid = train_df.dropna(subset=['future_max_return']).copy()
    mat = rule_matrix(valid)
    y = valid['future_max_return'] >= target
    rows = []
    names = list(mat.columns)
    base = float(y.mean()) if len(y) else np.nan
    for k in range(1, max_features + 1):
        for combo in combinations(names, k):
            mask = mat[list(combo)].all(axis=1)
            n = int(mask.sum())
            if n < min_signals:
                continue
            yy = y[mask]
            precision = float(yy.mean()) if len(yy) else np.nan
            lift = precision / base if base > 0 else np.nan
            rows.append({'特徵數':k,'訊號數':n,'達標數':int(yy.sum()),'Precision':precision,'BaseRate':base,'Lift':lift,'特徵組合':' + '.join(combo)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['Lift','Precision','訊號數'], ascending=[False,False,False]).head(top_n)


def apply_combo(df, combo_text):
    if not combo_text:
        return pd.Series(False, index=df.index)
    names = [z.strip() for z in combo_text.split(' + ') if z.strip()]
    mat = rule_matrix(df)
    if any(n not in mat.columns for n in names):
        return pd.Series(False, index=df.index)
    return mat[names].all(axis=1)


def metrics(y, pred):
    y = pd.Series(y).astype(bool)
    pred = pd.Series(pred).astype(bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    precision = tp/(tp+fp) if tp+fp else np.nan
    recall = tp/(tp+fn) if tp+fn else np.nan
    f1 = 2*precision*recall/(precision+recall) if pd.notna(precision) and pd.notna(recall) and precision+recall else np.nan
    specificity = tn/(tn+fp) if tn+fp else np.nan
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'Precision':precision,'Recall':recall,'F1':f1,'Specificity':specificity}


def walk_forward(df, target, forward_days, train_days, test_days, min_signals, max_features):
    valid = df.dropna(subset=['future_max_return']).copy()
    if len(valid) < train_days + test_days:
        return pd.DataFrame(), pd.DataFrame()
    rows=[]; predictions=[]
    start = train_days
    while start < len(valid):
        train = valid.iloc[start-train_days:start]
        test = valid.iloc[start:min(start+test_days, len(valid))]
        combos = search_combinations(train, target, min_signals, max_features, 1)
        if combos.empty:
            combo=''
        else:
            combo=combos.iloc[0]['特徵組合']
        pred=apply_combo(test, combo)
        m=metrics(test['future_max_return']>=target,pred)
        rows.append({'訓練結束':train.index[-1].date(),'測試開始':test.index[0].date(),'測試結束':test.index[-1].date(),'最佳組合':combo or '—','測試訊號':int(pred.sum()),'Precision':m['Precision'],'Recall':m['Recall'],'F1':m['F1'],'測試樣本':len(test)})
        q=test.copy(); q['WF訊號']=pred; q['WF組合']=combo
        predictions.append(q)
        start += test_days
    return pd.DataFrame(rows), pd.concat(predictions) if predictions else pd.DataFrame()


def similarity_score(hist, latest, features, topn=25):
    h = hist.dropna(subset=features).copy()
    if h.empty:
        return np.nan, pd.DataFrame()
    mat=h[features].astype(float); cur=latest[features].astype(float)
    std=mat.std().replace(0,1)
    h['距離']=np.sqrt(((mat-cur)/std).pow(2).sum(axis=1))
    sim=h.sort_values('距離').head(topn).copy()
    # 用歷史相似樣本的結果估算「模式命中」，不是直接宣稱未來機率
    hit=(sim['future_max_return']>=sim.attrs.get('target',0)).mean() if len(sim) else np.nan
    return hit, sim


def fmt_pct(v):
    return '—' if pd.isna(v) else f'{v:.1%}'


def confidence_score(oos_precision, similarity_hit, oos_n, similarity_n, match_strength):
    # 收縮估計：樣本越少，越向 50% 回歸；避免小樣本製造虛高信心。
    def shrink(p,n,k=25):
        if pd.isna(p): return 0.5
        return (p*n + 0.5*k)/(n+k)
    p=shrink(oos_precision,oos_n); s=shrink(similarity_hit,similarity_n)
    raw=0.45*p+0.35*s+0.20*match_strength
    return float(np.clip(100*raw,0,100))


def gated_signal(df, combo):
    dna=apply_combo(df,combo) if combo else pd.Series(False,index=df.index)
    tr=df.apply(lambda r: trend_score(r)[0],axis=1)
    vs=df.apply(lambda r: volume_pressure(r)[1],axis=1)
    cs=df.apply(lambda r: candle_score(r)[0],axis=1)
    gate=dna & (tr>=55) & (vs>=55) & (cs>=45)
    confirmed=gate & df['breakout_20'].fillna(False)
    return dna,tr,vs,cs,gate,confirmed


def decision_engine(x, train, oos, combo, target, min_conf=65, min_precision=0.55):
    latest=x.iloc[-1]
    dna=bool(apply_combo(x.tail(1),combo).iloc[0]) if combo else False
    _, oos_trend, oos_vol, oos_candle, oos_gate, oos_confirmed = gated_signal(oos, combo) if combo and len(oos) else (pd.Series(False,index=oos.index),pd.Series(False,index=oos.index),pd.Series(False,index=oos.index),pd.Series(False,index=oos.index),pd.Series(False,index=oos.index),pd.Series(False,index=oos.index))
    om=metrics(oos['future_max_return']>=target, oos_gate) if combo and len(oos) else metrics(pd.Series(dtype=bool),pd.Series(dtype=bool))
    features=['ret_5d','vol_ratio','macd_hist_change','qqq_5d','vix_5d','relative_strength_5d']
    hist=train.dropna(subset=features).copy()
    if len(hist):
        mat=hist[features].astype(float); cur=latest[features].astype(float); std=mat.std().replace(0,1)
        hist['距離']=np.sqrt(((mat-cur)/std).pow(2).sum(axis=1)); sim=hist.sort_values('距離').head(25).copy(); sim['達標']=sim['future_max_return']>=target
        sim_hit=float(sim['達標'].mean()) if len(sim) else np.nan
        q=float(hist['距離'].quantile(.25)); d=float(sim['距離'].iloc[0]) if len(sim) else np.inf
        match=float(np.clip(1-d/(3*q+1e-9),0,1)) if np.isfinite(d) else 0
    else: sim=pd.DataFrame(); sim_hit=np.nan; match=0
    trend,tparts=trend_score(latest); vlabel,vscore=volume_pressure(latest); cscore,clabels=candle_score(latest)
    breakout=bool(latest.get('breakout_20',False))
    dist=(float(latest['breakout_20_level'])/float(latest['Close'])-1) if pd.notna(latest.get('breakout_20_level')) else np.nan
    proximity=100 if pd.notna(dist) and dist<0 else max(0,100*(1-min(abs(dist)/.15,1))) if pd.notna(dist) else 50
    entry=.50*trend+.25*vscore+.25*cscore
    if pd.notna(dist) and dist>=0: entry=.80*entry+.20*proximity
    strong_setup=dna and match>=.35 and pd.notna(om['Precision']) and om['Precision']>=min_precision
    bearish=trend<45 or (trend<55 and vscore<40) or cscore<35
    confirmed=breakout and trend>=55 and vscore>=55 and cscore>=45
    if strong_setup and confirmed and entry>=55: direction='🟢 LONG'
    elif strong_setup and not bearish and trend>=50 and entry>=50: direction='🟡 EARLY SETUP'
    elif strong_setup: direction='🟡 WAIT'
    else: direction='⚪ NO TRADE'
    def shrink(p,n,k=25): return .5 if pd.isna(p) else (p*n+.5*k)/(n+k)
    conf=float(np.clip(100*(.38*shrink(om['Precision'],int(oos_gate.sum()) if combo else 0)+.25*shrink(sim_hit,len(sim))+.17*match+.12*trend/100+.08*entry/100),0,100))
    return {'direction':direction,'confidence':conf,'oos':om,'similarity_hit':sim_hit,'similarity_n':len(sim),'match_strength':match,'sim':sim,'trend':trend,'trend_parts':tparts,'volume_label':vlabel,'volume_score':vscore,'candle_score':cscore,'candle_labels':clabels,'entry_quality':entry,'current_dna':dna,'strong_setup':strong_setup,'breakout_confirmed':confirmed,'bearish_trend':bearish,'breakout_level':float(latest['breakout_20_level']) if pd.notna(latest.get('breakout_20_level')) else float(latest['Close'])}


def trade_plan(x, decision, rr1=1.5, rr2=2.5):
    r=x.iloc[-1]; price=float(r['Close']); atr=float(r['atr']) if pd.notna(r['atr']) else price*.03
    breakout=float(r['breakout_20_level']) if pd.notna(r.get('breakout_20_level')) else price+atr
    swing=min(float(r['swing_low_10']) if pd.notna(r.get('swing_low_10')) else price-1.5*atr,float(r['swing_low_20']) if pd.notna(r.get('swing_low_20')) else price-1.5*atr)
    stop=min(swing,price-1.25*atr)
    early_low=max(.01,price-.40*atr); early_high=price+.35*atr
    conf_low=breakout; conf_high=breakout+.50*atr
    risk_e=max(price-stop,.01); risk_c=max(conf_low-stop,.01)
    return {'Current':price,'Early Low':early_low,'Early High':early_high,'Breakout':breakout,'Confirmation Low':conf_low,'Confirmation High':conf_high,'Stop':stop,'T1 Early':price+1.8*risk_e,'T2 Early':price+3*risk_e,'T1 Confirm':conf_low+1.5*risk_c,'T2 Confirm':conf_low+2.5*risk_c,'Risk Early':risk_e/price,'Risk Confirm':risk_c/conf_low}

# =========================
# UI
# =========================
st.title('🔥 股票爆升前決策引擎 V2.4')
st.caption('第一眼：交易方向；第二眼：原因；第三眼：證據。每隻股票獨立建立 DNA，並以 OOS / Walk-Forward 驗證。')

with st.sidebar:
    st.header('⚙️ 研究設定')
    symbol=st.text_input('股票代號','RKLB').upper().strip()
    period=st.selectbox('歷史資料',['5y','10y'],index=0)
    st.caption('V2.4 已移除大部分手動參數。爆升定義、DNA及 OOS 設定由系統自動搜尋；詳情放在頁面底部。')

if not symbol: st.stop()
df=load_data(symbol,period); qqq,vix=load_macro(period)
if df.empty:
    st.error('找不到股票資料，請檢查股票代號。'); st.stop()
# V2.4 自動選擇爆升定義：只使用前70%資料，OOS完全不參與。
base=add_v24_features(add_features(df,qqq,vix))
pre_n=max(100,int(len(base)*0.70)); pre=base.iloc[:pre_n].copy()
label_candidates=[(5,.05),(10,.10),(10,.15),(20,.10),(20,.15),(20,.20),(30,.15),(30,.20),(30,.30)]
label_rows=[]
for fw,tg in label_candidates:
    zz=make_labels(pre,fw,tg,10); vv=zz.dropna(subset=['future_max_return'])
    if len(vv)>=80:
        rate=float(vv.target_hit.mean()); events=int(vv.explosion_event.sum())
        if 0.03<=rate<=0.60 and events>=8:
            label_rows.append({'Forward':fw,'Target':tg,'BaseRate':rate,'Events':events,'Score':(1-abs(rate-.18))*np.log1p(events)})
labels_df=pd.DataFrame(label_rows)
if labels_df.empty: forward,target=20,.15
else:
    best_label=labels_df.sort_values('Score',ascending=False).iloc[0]
    forward,target=int(best_label.Forward),float(best_label.Target)
x=make_labels(base,forward,target,10)
valid=x.dropna(subset=['future_max_return']).copy()
sp=max(80,int(len(valid)*.70)); train=valid.iloc[:sp].copy(); oos=valid.iloc[sp:].copy()
min_signals=max(10,int(len(train)*.015)); max_features=3; min_conf=65; min_precision=.55; cooldown=10
combos=search_combinations(train,target,min_signals,max_features,30)
best_combo=combos.iloc[0]['特徵組合'] if not combos.empty else ''
dec=decision_engine(x,train,oos,best_combo,target,min_conf,min_precision)
plan=trade_plan(x,dec)
latest=x.iloc[-1]
# =========================
# 第一層：直接答案
# =========================
st.markdown('## 🔥 TODAY’S SIGNAL')
c1,c2,c3,c4,c5=st.columns(5)
c1.metric('股票',symbol); c2.metric('交易狀態',dec['direction']); c3.metric('Model Confidence',f"{dec['confidence']:.0f}/100"); c4.metric('OOS Precision',fmt_pct(dec['oos']['Precision'])); c5.metric('Historical Match',f"{dec['match_strength']:.0%}")
if dec['direction']=='🟢 LONG': st.success(f"**{symbol} → 🟢 LONG**\n\n爆升 DNA、Trend Gate、成交量方向及 K 線結構同時支持多頭，且突破條件已確認。")
elif dec['direction']=='🟡 EARLY SETUP': st.warning(f"**{symbol} → 🟡 EARLY SETUP**\n\n爆升 DNA 已出現，趨勢沒有明顯偏空，但尚未完成突破確認。")
elif dec['direction']=='🟡 WAIT': st.warning(f"**{symbol} → 🟡 WAIT**\n\n爆升 DNA 存在，但目前 Trend Gate / Price Action 不足以支持直接 LONG。")
else: st.info(f"**{symbol} → ⚪ NO TRADE**\n\n目前沒有足夠的自身歷史統計優勢。")

st.markdown('## 🔎 WHY — 為什麼現在是這個訊號？')
w1,w2,w3,w4=st.columns(4); w1.metric('🚀 爆升 DNA','強' if dec['strong_setup'] else '不足'); w2.metric('📈 Trend Gate',f"{dec['trend']:.0f}/100"); w3.metric('💰 Entry Quality',f"{dec['entry_quality']:.0f}/100"); w4.metric('📦 Volume Pressure',dec['volume_label'])

st.markdown('### ① 歷史狀態：現在像不像 RKLB 自己過去爆升前？')
sim=dec['sim']
if sim.empty: st.info('沒有足夠相似案例。')
else:
    a,b,c,d=st.columns(4); a.metric('相似案例',len(sim)); b.metric(f'達 +{target:.0%}',fmt_pct((sim.future_max_return>=target).mean())); c.metric('達 +10%',fmt_pct((sim.future_max_return>=.10).mean())); d.metric('達 +20%',fmt_pct((sim.future_max_return>=.20).mean()))
    st.write(f"目前狀態與 {symbol} 過去相似狀態的匹配度約 **{dec['match_strength']:.0%}**。這是歷史相似度，不是未來上升機率。")

st.markdown('### ② Trend Gate：趨勢是否支持現在買？')
tdf=pd.DataFrame([{'條件':k,'結果':'🟢' if v else '🔴'} for k,v in dec['trend_parts'].items()]); st.dataframe(tdf,use_container_width=True,hide_index=True)
if dec['trend']<45: st.error('🔴 **趨勢偏空：即使爆升 DNA 很強，系統也禁止直接 LONG。**')
elif dec['trend']<60: st.warning('🟡 趨勢尚未完全轉強，系統偏向 WAIT。')
else: st.success('🟢 趨勢結構支持多頭。')

st.markdown('### ③ Volume + Price Action：放量是買盤還是賣壓？')
v1,v2,v3=st.columns(3); v1.metric('Volume Ratio',f"{latest['vol_ratio']:.2f}×" if pd.notna(latest['vol_ratio']) else '—'); v2.metric('單日價格變化',fmt_pct(latest['ret_1d'])); v3.metric('Volume Pressure',dec['volume_label'])
if dec['volume_score']<40: st.error('🔴 **放量下跌 = 賣壓。V2.4 不再把「Volume Ratio 高」自動當成 bullish。**')
elif dec['volume_score']>=70: st.success('🟢 放量與價格方向一致，偏向買盤確認。')
else: st.info('⚪ 成交量方向沒有強烈確認。')

st.markdown('### ④ K線結構')
st.write('、'.join(dec['candle_labels']) if dec['candle_labels'] else '沒有觸發主要特殊 K 線形態。'); st.progress(dec['candle_score']/100); st.caption(f"K線 Price Action Score：{dec['candle_score']:.0f}/100")

st.markdown('### ⑤ ⚠️ Counter-Evidence')
risks=[]
if dec['trend']<55: risks.append(f"Trend Gate 只有 {dec['trend']:.0f}/100，價格結構仍偏弱。")
if dec['volume_score']<50: risks.append(f"{dec['volume_label']}，不能把成交量增加直接解讀為利多。")
if dec['candle_score']<45: risks.append('最近 K 線沒有提供足夠買方確認。')
if not bool(latest.get('breakout_20',False)): risks.append(f"尚未突破前20日高位 ${dec['breakout_level']:,.2f}。")
for r in risks or ['目前沒有主要反向證據，但仍需留意 OOS 樣本量。']: st.write('⚠️ '+r)

st.markdown('## 📍 TRADE PLAN')
price=plan['Current']
rows=[
 {'策略':'🟡 Early Entry','進場區':f"${plan['Early Low']:,.2f} – ${plan['Early High']:,.2f}",'目前是否適用':'❌ 否，Trend Gate 偏空' if dec['bearish_trend'] else '🟡 可研究','Breakout':f"${plan['Breakout']:,.2f}",'Stop':f"${plan['Stop']:,.2f}",'Target 1':f"${plan['T1 Early']:,.2f}",'Target 2':f"${plan['T2 Early']:,.2f}",'風險':fmt_pct(plan['Risk Early'])},
 {'策略':'🟢 Confirmation Entry','進場區':f"${plan['Confirmation Low']:,.2f} – ${plan['Confirmation High']:,.2f}",'目前是否適用':'🟢 已確認' if dec['breakout_confirmed'] else '🟡 等待突破','Breakout':f"${plan['Breakout']:,.2f}",'Stop':f"${plan['Stop']:,.2f}",'Target 1':f"${plan['T1 Confirm']:,.2f}",'Target 2':f"${plan['T2 Confirm']:,.2f}",'風險':fmt_pct(plan['Risk Confirm'])}
]
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
p1,p2,p3=st.columns(3); p1.metric('Current Price',f"${price:,.2f}"); p2.metric('Breakout',f"${plan['Breakout']:,.2f}"); p3.metric('Invalidation',f"${plan['Stop']:,.2f}")

st.markdown('### 🧭 三種劇本')
sc=pd.DataFrame([
 ['🟢 Breakout確認',f">= ${plan['Breakout']:,.2f}",'收市突破 + 成交量/Price Action確認','再評估 LONG'],
 ['🟡 目前狀態',f"${price:,.2f}",f"Trend {dec['trend']:.0f} / Volume {dec['volume_score']:.0f} / Candle {dec['candle_score']:.0f}",'若趨勢未改善 → WAIT'],
 ['🔴 結構失效',f"< ${plan['Stop']:,.2f}",'跌破風險界線/近期結構低點','取消原本爆升假設']
],columns=['劇本','價格條件','確認條件','系統行動']); st.dataframe(sc,use_container_width=True,hide_index=True)
st.caption('Entry / Stop / Target 是研究型風險管理計算，不代表保證成交或未來價格。')

# =========================
# 第二層：證據
# =========================
T1,T2,T3,T4,T5,T6,T7,T8=st.tabs(['📊 歷史證據','🧬 股票 DNA','🧪 Walk-Forward','🧩 相似案例','🌡️ Regime','❌ 成功/失敗','🔎 Scanner','📚 研究原則'])

with T1:
    st.subheader('歷史證據')
    a,b,c,d,e=st.columns(5)
    a.metric('歷史爆升事件',int(x['explosion_event'].sum()))
    b.metric('Train 樣本',len(train)); c.metric('OOS 樣本',len(oos))
    d.metric('OOS Recall',fmt_pct(dec['oos']['Recall'])); e.metric('OOS F1',fmt_pct(dec['oos']['F1']))
    if not oos.empty and best_combo:
        op=gated_signal(oos,best_combo)[4]; hit=oos.loc[op,'future_max_return']
        st.write(f"OOS Gate 訊號：**{int(op.sum())}**；其中達到 +{target:.0%}：**{int((hit>=target).sum())}**")
        tmp=oos.loc[op,['Close','future_max_return','future_close_return','future_min_return']].tail(100).copy(); st.dataframe(tmp,use_container_width=True)
    st.line_chart(x['Close'])

with T2:
    st.subheader(f'🧬 {symbol} 股票 DNA')
    st.write('這裡只描述這隻股票自己的歷史特徵，不把另一隻股票的勝率直接套過來。')
    if combos.empty: st.warning('目前樣本不足以找到穩定特徵組合。')
    else:
        show=combos.copy(); show['Precision']=show['Precision'].map(fmt_pct); show['BaseRate']=show['BaseRate'].map(fmt_pct); show['Lift']=show['Lift'].map(lambda z:'—' if pd.isna(z) else f'{z:.2f}x')
        st.dataframe(show,use_container_width=True,hide_index=True)
        st.success(f'目前最終 DNA 組合：**{best_combo}**')
        st.write(f"目前最新日符合 DNA：**{'是' if bool(apply_combo(x.tail(1),best_combo).iloc[0]) else '否'}**")

with T3:
    st.subheader('Walk-Forward：每一段都重新學習 DNA，再到下一段 OOS')
    wf_train=min(504,max(252,int(len(valid)*.45))); wf_test=126
    wf, pred=walk_forward(x,target,wf_train,wf_test,min_signals,max_features)
    if wf.empty: st.warning('資料不足以完成目前 Walk-Forward 設定，請增加歷史資料或縮短訓練年數。')
    else:
        st.dataframe(wf,use_container_width=True,hide_index=True)
        st.metric('Walk-Forward 平均 OOS Precision',fmt_pct(wf['Precision'].mean()))
        st.caption('每個測試區間的最佳組合只使用更早的訓練資料搜尋；測試區間不回饋參數。')

with T4:
    st.subheader('目前狀態 vs 自己過去的相似狀態')
    features=['ret_5d','vol_ratio','macd_hist_change','qqq_5d','vix_5d','relative_strength_5d']
    sim=dec['sim'].copy()
    if sim.empty: st.info('沒有足夠相似案例。')
    else:
        sim['結果']=np.where(sim['future_max_return']>=target,'達標','未達標')
        st.dataframe(sim[['Close']+features+['future_max_return','結果','距離']],use_container_width=True)

with T5:
    st.subheader('市場 Regime')
    z=valid.dropna(subset=['vix_level','ret_20d']).copy()
    if z.empty: st.info('沒有足夠資料。')
    else:
        med=z['vix_level'].median()
        z['市場環境']=np.select([(z.vix_level<=med)&(z.ret_20d>=0),(z.vix_level>med)&(z.ret_20d>=0),(z.vix_level<=med)&(z.ret_20d<0)],['低VIX＋上升','高VIX＋上升','低VIX＋下跌'],default='高VIX＋下跌')
        if best_combo:
            z['訊號']=apply_combo(z,best_combo); z['達標']=z.future_max_return>=target
            rows=[]
            for name,g in z.groupby('市場環境'):
                s=g[g.訊號]; rows.append({'市場環境':name,'樣本':len(g),'訊號':len(s),'Precision':s.達標.mean() if len(s) else np.nan,'BaseRate':g.達標.mean()})
            rt=pd.DataFrame(rows); rt['Precision']=rt.Precision.map(fmt_pct); rt['BaseRate']=rt.BaseRate.map(fmt_pct); st.dataframe(rt,use_container_width=True,hide_index=True)

with T6:
    st.subheader('成功 / 失敗：檢查模型是不是只對少數案例有效')
    s=valid[valid.score if 'score' in valid else valid.index] if False else valid.copy()
    if best_combo:
        s['訊號']=apply_combo(s,best_combo); s=s[s.訊號]
        if len(s):
            s['成功']=s.future_max_return>=target
            cols=['ret_5d','vol_ratio','macd_hist_change','qqq_5d','vix_5d','relative_strength_5d','future_max_return','future_min_return']
            st.dataframe(s.groupby('成功')[cols].mean(),use_container_width=True)
            st.write(f'歷史 DNA 訊號：**{len(s)}**；達標：**{int(s.成功.sum())}**。')
        else: st.info('沒有歷史 DNA 訊號。')

with T7:
    st.subheader('🔎 多股票 Scanner：每隻股票獨立 DNA + Trend Gate')
    tickers=st.text_area('股票代號（逗號分隔）','TSLA,NVDA,AAPL,AMD,AVGO,MU,PLTR,GOOG,AMZN,RKLB').upper()
    if st.button('開始掃描 V2.4'):
        rows=[]
        for s in [q.strip() for q in tickers.split(',') if q.strip()]:
            try:
                d=load_data(s,period)
                if d.empty: continue
                zz=add_v24_features(add_features(d,qqq,vix))
                # 每隻股票自己選 label，只用自己的前70%。
                pn=max(100,int(len(zz)*.70)); pre_s=zz.iloc[:pn]
                lr=[]
                for fw,tg in label_candidates:
                    tmp=make_labels(pre_s,fw,tg,10); vv=tmp.dropna(subset=['future_max_return'])
                    if len(vv)>=80:
                        rate=float(vv.target_hit.mean()); events=int(vv.explosion_event.sum())
                        if .03<=rate<=.60 and events>=8: lr.append((fw,tg,(1-abs(rate-.18))*np.log1p(events)))
                if lr: f,t,_=sorted(lr,key=lambda q:q[2],reverse=True)[0]
                else: f,t=20,.15
                zz=make_labels(zz,f,t,10); vv=zz.dropna(subset=['future_max_return']); ss=max(80,int(len(vv)*.70)); tr=vv.iloc[:ss]; oo=vv.iloc[ss:]
                cr=search_combinations(tr,t,max(10,int(len(tr)*.015)),3,30); bc=cr.iloc[0]['特徵組合'] if not cr.empty else ''
                de=decision_engine(zz,tr,oo,bc,t,min_conf,min_precision)
                rows.append({'股票':s,'訊號':de['direction'],'Confidence':de['confidence'],'OOS Precision':de['oos']['Precision'],'Historical Match':de['match_strength'],'Trend':de['trend'],'Volume':de['volume_label'],'最新價':zz.iloc[-1]['Close'],'DNA':bc or '—'})
            except Exception as ex:
                st.warning(f'{s} 失敗：{ex}')
        if rows:
            out=pd.DataFrame(rows).sort_values('Confidence',ascending=False); out['Confidence']=out.Confidence.map(lambda v:'—' if pd.isna(v) else f'{v:.0f}/100'); out['OOS Precision']=out['OOS Precision'].map(fmt_pct); out['Historical Match']=out['Historical Match'].map(fmt_pct); out['Trend']=out['Trend'].map(lambda v:'—' if pd.isna(v) else f'{v:.0f}'); st.dataframe(out,use_container_width=True,hide_index=True)

with T8:
    st.subheader('研究原則')
    st.markdown('''
- **先方向、後證據、再細節**：首頁先給 LONG / WAIT / NO TRADE，再展示原因與完整資料。
- **每股獨立 DNA**：TSLA 的歷史特徵不直接套到 NVDA；每隻股票重新搜尋自己的特徵組合。
- **真正 OOS**：最終模型的 OOS 資料不參與最佳組合搜尋。
- **Walk-Forward**：每一個時間窗口重新訓練，下一段才測試。
- **Confidence 不是保證機率**：它是由 OOS Precision、相似歷史案例、樣本量收縮及相似度組成的研究型可信度分數。
- **Trade Plan 是計算結果**：Entry / Stop / Target 使用 ATR、突破位及 swing low 等資料生成；實際成交會受滑價、跳空及流動性影響。
- **沒有優勢就 NO TRADE**：不為了每天產生訊號而強行交易。
- **目前只做上行爆升方向模型**：SHORT 需要另外建立「下跌/崩跌前」標籤與獨立 OOS 模型，不能直接把 LONG 模型反轉。
''')


with st.expander('🔬 自動選參數（一般使用者不用修改）'):
    st.write(f'Forward Days：**{forward}**；Target：**+{target:.0%}**；Train/OOS：**70% / 30%**')
    st.write(f'最終 DNA：**{best_combo or "沒有穩定組合"}**；最少訊號數：**{min_signals}**')
    if labels_df.empty:
        st.info('沒有足夠資料比較不同爆升定義。')
    else:
        show=labels_df.copy(); show['BaseRate']=show['BaseRate'].map(fmt_pct); st.dataframe(show,use_container_width=True,hide_index=True)
