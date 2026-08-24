import datetime
import json
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==================== 监控股票池 ====================
WATCHLIST = [
    "TSLA",
    "NVDA",
    "AAPL",
    "MSFT",
    "AMD",
    "AMZN",
    "META",
    "GOOGL",
    "NFLX",
    "SPCX",
    "AVGO",
    "PLTR",
    "TSM",
    "CRWV",
    "QQQ",
    "SPY",
]

# 从 GitHub Secrets 中读取 Discord Webhook 环境变量
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")


def calculate_macd_and_ema(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MACD 与 250 EMA"""
    df = df.copy()
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["HIST"] = (df["DIF"] - df["DEA"]) * 2
    df["EMA_250"] = df["Close"].ewm(span=250, adjust=False).mean()
    return df


def segment_macd_waves(df: pd.DataFrame) -> List[Dict]:
    """将 MACD 直方图按正负切分为红/绿柱堆，并积分计算做功面积"""
    waves = []
    if len(df) < 20:
        return waves
    df = df.copy()
    # 美股规范: HIST < 0 为 0 轴下方红柱 (空头做功)
    df["sign"] = np.where(df["HIST"] < 0, -1, 1)
    df["group"] = (df["sign"] != df["sign"].shift()).cumsum()

    for _, group_df in df.groupby("group"):
        wave_type = "red" if group_df["sign"].iloc[0] == -1 else "green"
        area = group_df["HIST"].abs().sum()  # 柱体绝对值面积积分
        if wave_type == "red":
            extreme_dif = group_df["DIF"].min()
            extreme_price = group_df["Low"].min()
        else:
            extreme_dif = group_df["DIF"].max()
            extreme_price = group_df["High"].max()

        waves.append(
            {
                "type": wave_type,
                "area": float(area),
                "extreme_dif": float(extreme_dif),
                "extreme_price": float(extreme_price),
            }
        )
    return waves


def send_discord_card(data: dict):
    """向 Discord 发送结构化买点卡片"""
    if not DISCORD_WEBHOOK:
        print("未配置 DISCORD_WEBHOOK，跳过推送")
        return

    embed = {
        "title": f"🚀【1h MACD 面积背离买点 T1】 - {data['ticker']}",
        "color": 0x2ECC71,  # 绿色高亮边框
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
        "footer": {"text": "MACD 1h Trading Agent • 包含美股盘前盘后夜盘数据"},
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    payload = {
        "username": "MACD Trading Agent",
        "avatar_url": "https://i.imgur.com/4M34hi2.png",
        "embeds": [embed],
    }

    try:
        resp = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        if resp.status_code in [200, 204]:
            print(f"[{data['ticker']}] Discord 推送成功！")
        else:
            print(f"Discord 推送异常: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Discord 推送请求失败: {e}")


def main():
    print(
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始扫描标的池 ({len(WATCHLIST)} 只)..."
    )

    for ticker in WATCHLIST:
        try:
            # 1h 级别 K 线，prepost=True 开启盘前盘后/夜盘数据
            df_1h = yf.download(
                ticker,
                period="60d",
                interval="1h",
                prepost=True,
                progress=False,
            )
            if df_1h.empty:
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

            # 手稿核心条件判定：
            # 1. 价格破前低: P2 < P1
            # 2. DIF 抬高底背离: A2 > A1
            # 3. 面积萎缩过半: B2 < 0.5 * B1
            if P2 < P1 and A2 > A1 and B2 < (0.5 * B1):
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
                send_discord_card(signal_data)

        except Exception as e:
            print(f"扫描 {ticker} 发生错误: {e}")

    print("本轮扫描完成。\n")


if __name__ == "__main__":
    main()
