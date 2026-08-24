import datetime
import json
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==================== 1. 去重合并后的自选股票池 (19只) ====================
WATCHLIST = [
    "TSLA",
    "NVDA",
    "AAPL",
    "MSFT",
    "META",
    "GOOGL",
    "INTC",
    "IBM",
    "CRWV",
    "RKLB",
    "AVGO",
    "PLTR",
    "QQQ",
    "SPY",
    "AMD",
    "AMZN",
    "NFLX",
    "SPCX",
    "TSM",
]

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
STATE_FILE = "sent_alerts.json"


# ==================== 2. 状态记忆与防重复模块 ====================
def load_sent_alerts() -> dict:
    """加载已推送记录"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_sent_alerts(alerts: dict):
    """保存已推送记录到云端文件"""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存状态文件失败: {e}")


# ==================== 3. 指标与波浪切分算法 ====================
def calculate_macd_and_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["HIST"] = (df["DIF"] - df["DEA"]) * 2
    df["EMA_250"] = df["Close"].ewm(span=250, adjust=False).mean()
    return df


def segment_macd_waves(df: pd.DataFrame) -> List[Dict]:
    waves = []
    if len(df) < 20:
        return waves
    df = df.copy()
    # 美股规范: HIST < 0 为 0 轴下方红柱 (空头做功)
    df["sign"] = np.where(df["HIST"] < 0, -1, 1)
    df["group"] = (df["sign"] != df["sign"].shift()).cumsum()

    for _, group_df in df.groupby("group"):
        wave_type = "red" if group_df["sign"].iloc[0] == -1 else "green"
        area = group_df["HIST"].abs().sum()  # 柱体面积积分
        if wave_type == "red":
            extreme_dif = group_df["DIF"].min()
            extreme_price = group_df["Low"].min()
        else:
            extreme_dif = group_df["DIF"].max()
            extreme_price = group_df["High"].max()

        waves.append(
            {
                "type": wave_type,
                "start_time": str(group_df.index[0]),
                "end_time": str(group_df.index[-1]),
                "area": float(area),
                "extreme_dif": float(extreme_dif),
                "extreme_price": float(extreme_price),
                "bars": len(group_df),
            }
        )
    return waves


# ==================== 4. Discord 卡片推送 ====================
def send_discord_card(data: dict) -> bool:
    if not DISCORD_WEBHOOK:
        print("未配置 DISCORD_WEBHOOK")
        return False

    embed = {
        "title": f"🚀【1h MACD 面积背离买点 T1】 - {data['ticker']}",
        "color": 0x2ECC71,
        "description": f"**标的代码**: `{data['ticker']}` | **当前价**: `${data['current_price']:.2f}`",
        "fields": [
            {
                "name": "📉 价格形态 (创新低破底)",
                "value": f"P1: `${data['P1']:.2f}` ➔ P2: `${data['P2']:.2f}`",
                "inline": True,
            },
            {
                "name": "⚡ DIF 指标 (底背离抬高)",
                "value": f"A1: `{data['A1']:.4f}` ➔ A2: `{data['A2']:.4f}`",
                "inline": True,
            },
            {
                "name": "📊 面积衰竭 (B2 < 0.5 B1)",
                "value": f"**{data['ratio']:.1%}** (B1: `{data['B1']:.1f}` / B2: `{data['B2']:.1f}`)",
                "inline": False,
            },
            {
                "name": "🎯 第一止盈目标 (250 EMA)",
                "value": f"**${data['target_ema250']:.2f}**",
                "inline": True,
            },
            {
                "name": "🛑 建议止损位 (-5% ~ -10%)",
                "value": f"${data['sl_5']:.2f} ~ ${data['sl_10']:.2f}",
                "inline": True,
            },
        ],
        "footer": {"text": "MACD 1h Agent • 单次买点精准推送 (已过滤重复)"},
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    payload = {
        "username": "MACD Trading Bot",
        "avatar_url": "https://i.imgur.com/4M34hi2.png",
        "embeds": [embed],
    }

    try:
        resp = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        return resp.status_code in [200, 204]
    except Exception as e:
        print(f"Discord 推送失败: {e}")
        return False


# ==================== 5. 主扫描逻辑 ====================
def main():
    sent_alerts = load_sent_alerts()
    has_new_alert = False

    print(
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始扫描 {len(WATCHLIST)} 只标的..."
    )

    for ticker in WATCHLIST:
        try:
            df_1h = yf.download(
                ticker,
                period="60d",
                interval="1h",
                prepost=True,
                progress=False,
            )
            if df_1h.empty or len(df_1h) < 30:
                continue
            if isinstance(df_1h.columns, pd.MultiIndex):
                df_1h.columns = df_1h.columns.get_level_values(0)

            df_1h = calculate_macd_and_ema(df_1h)
            waves = segment_macd_waves(df_1h)
            red_waves = [w for w in waves if w["type"] == "red"]

            if len(red_waves) < 2:
                continue

            w1, w2 = red_waves[-2], red_waves[-1]
            P1, A1, B1 = w1["extreme_price"], w1["extreme_dif"], w1["area"]
            P2, A2, B2 = w2["extreme_price"], w2["extreme_dif"], w2["area"]

            # 1. 核心底背离与面积衰竭条件
            cond_divergence = (P2 < P1) and (A2 > A1) and (B2 < (0.5 * B1))

            # 2. 拐点触发条件 (红柱处于当前收缩/拐头阶段)
            h_curr = df_1h["HIST"].iloc[-1]
            h_prev = df_1h["HIST"].iloc[-2]
            cond_inflection = h_curr > h_prev  # 柱体向0轴收敛

            if cond_divergence and cond_inflection:
                # 唯一波段指纹 ID (由标的代码 + 第一波结束时间 + 第二波开始时间 唯一锁定)
                signal_unique_id = (
                    f"{ticker}_{w1['end_time']}_{w2['start_time']}"
                )

                # 如果这个波段已经推送过，直接跳过！绝不重复推送
                if signal_unique_id in sent_alerts:
                    print(
                        f"[{ticker}] 信号已于 {sent_alerts[signal_unique_id]} 推送过，跳过重复通知。"
                    )
                    continue

                latest_close = float(df_1h["Close"].iloc[-1])
                target_ema250 = float(df_1h["EMA_250"].iloc[-1])

                signal_data = {
                    "ticker": ticker,
                    "current_price": latest_close,
                    "P1": P1,
                    "P2": P2,
                    "A1": A1,
                    "A2": A2,
                    "B1": B1,
                    "B2": B2,
                    "ratio": B2 / B1,
                    "target_ema250": target_ema250,
                    "sl_5": latest_close * 0.95,
                    "sl_10": latest_close * 0.90,
                }

                if send_discord_card(signal_data):
                    print(f"[{ticker}] 🚀 新买点推送成功！")
                    sent_alerts[signal_unique_id] = (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    has_new_alert = True

        except Exception as e:
            print(f"扫描 {ticker} 发生错误: {e}")

    # 若有新推送，更新状态文件
    if has_new_alert:
        save_sent_alerts(sent_alerts)

    print("本轮扫描完成。\n")


if __name__ == "__main__":
    main()
