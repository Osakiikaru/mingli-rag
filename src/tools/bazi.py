from datetime import date, timedelta

from langchain_core.tools import tool
from lunar_python import Solar


def get_bazi(year: int, month: int, day: int, hour: int, gender: str) -> dict:
    """
    输入公历生辰，返回完整八字信息（含大运）。
    hour: 0-23 整数。23点为夜子时，日柱归次日，时柱仍为子时。
    gender: "男" or "女"
    """
    # 夜子时（23:00-24:00）：日柱归次日，时柱与次日早子时相同
    if hour == 23:
        next_day = date(year, month, day) + timedelta(days=1)
        year, month, day, hour = next_day.year, next_day.month, next_day.day, 0

    solar = Solar.fromYmdHms(year, month, day, hour, 0, 0)
    lunar = solar.getLunar()
    bazi  = lunar.getEightChar()

    # ── 四柱基础信息 ─────────────────────────────────────────
    result = {
        "年柱": {"天干": bazi.getYearGan(),  "地支": bazi.getYearZhi()},
        "月柱": {"天干": bazi.getMonthGan(), "地支": bazi.getMonthZhi()},
        "日柱": {"天干": bazi.getDayGan(),   "地支": bazi.getDayZhi()},
        "时柱": {"天干": bazi.getTimeGan(),  "地支": bazi.getTimeZhi()},
        "性别": gender,
        "日元": bazi.getDayGan(),
    }

    # ── 大运信息 ─────────────────────────────────────────────
    try:
        gender_int = 1 if gender == "男" else 0
        yun        = bazi.getYun(gender_int, sect=1)  # sect=1: 天数计算法

        # 起运时间
        start_y = yun.getStartYear()
        start_m = yun.getStartMonth()
        start_d = yun.getStartDay()

        # 起运描述：几岁几月几天
        start_parts = []
        if start_y: start_parts.append(f"{start_y}岁")
        if start_m: start_parts.append(f"{start_m}月")
        if start_d: start_parts.append(f"{start_d}天")
        start_str = "".join(start_parts) if start_parts else "0岁"

        forward_str = "顺推" if yun.isForward() else "逆推"

        # 大运列表（取 8 步，跳过 index=0 的行运前占位）
        da_yun_raw  = yun.getDaYun(9)   # 多取一个，去掉占位
        da_yun_list = []
        for dy in da_yun_raw:
            if dy.getIndex() == 0:      # index=0 是行运前，跳过
                continue
            da_yun_list.append({
                "干支":   dy.getGanZhi(),
                "起运岁": dy.getStartAge(),
                "结束岁": dy.getEndAge(),
                "起运年": dy.getStartYear(),
                "结束年": dy.getEndYear(),
            })
            if len(da_yun_list) >= 8:
                break

        result["起运"] = start_str
        result["顺逆"] = forward_str
        result["大运列表"] = da_yun_list

    except Exception as e:
        result["起运"]   = "未知"
        result["顺逆"]   = "未知"
        result["大运列表"] = []

    return result


@tool
def bazi_calculator(year: int, month: int, day: int, hour: int, gender: str) -> str:
    """计算八字四柱及大运，输入公历生日和性别，返回准确的年月日时四柱和大运信息"""
    result = get_bazi(year, month, day, hour, gender)
    return str(result)
