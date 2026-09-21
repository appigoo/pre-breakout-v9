# 股票爆升前決策引擎 V2.4

## 核心改動
V2.4 將 **Signal、Entry、Breakout** 分開，不再因為「爆升前 DNA」強就直接把目前價格標成買入。

流程：
`Stock DNA → Historical Match → Trend Gate → Volume Direction → K線 Gate → Entry Quality → LONG / EARLY SETUP / WAIT / NO TRADE`

### 新增
- Trend Gate：EMA20/50/200、均線斜率、Higher Low / Lower High、相對 QQQ
- Volume Direction：區分放量上升與放量下跌
- K線 Gate：Bearish/Bullish Engulfing、放量大陰/陽燭、收市位置
- Entry Quality：與爆升 DNA 分開
- Early Entry / Confirmation Entry 分開
- Breakout / WAIT / Invalidation 三種劇本
- WHY：加入歷史相似案例、趨勢、成交量方向、K線及 Counter-Evidence
- 自動選擇爆升定義；只使用 Train，不用 OOS 選參數
- OOS Precision 以 Trend/Volume/Candle Gate 後的訊號驗證
- 每隻股票獨立 Stock DNA
- Scanner 使用每隻股票自己的 DNA + Trend Gate
- 左側只保留股票代號及歷史資料年期

## 執行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 注意
這是研究型統計工具。Confidence 不是未來上升機率，也不是保證。實際交易仍有滑價、跳空、流動性及模型失效風險。
