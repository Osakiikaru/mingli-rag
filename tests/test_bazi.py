import pytest

from src.tools.bazi import get_bazi

TIAN_GAN = set("甲乙丙丁戊己庚辛壬癸")
DI_ZHI   = set("子丑寅卯辰巳午未申酉戌亥")

def assert_valid_bazi(result):
    for pillar in ["年柱", "月柱", "日柱", "时柱"]:
        assert result[pillar]["天干"] in TIAN_GAN
        assert result[pillar]["地支"] in DI_ZHI
    assert result["日元"] in TIAN_GAN

# ── Case 1：正常案例 —— 毛泽东 1893-12-26 子时 ──────────────────────────────
def test_normal_mao():
    # 1893年为癸巳年，年柱天干应为癸
    result = get_bazi(1893, 12, 26, 0, "男")
    assert result["年柱"]["天干"] == "癸"
    assert result["年柱"]["地支"] == "巳"
    assert all(k in result for k in ["年柱", "月柱", "日柱", "时柱", "日元"])

# ── Case 2：正常案例 —— 李嘉诚 1928-07-29 辰时 ──────────────────────────────
def test_normal_li():
    # 1928年为戊辰年，年柱天干应为戊
    result = get_bazi(1928, 7, 29, 8, "男")
    assert result["年柱"]["天干"] == "戊"
    assert result["年柱"]["地支"] == "辰"

# ── Case 3：立春边界 ──────────────────────────────────────────────────────────
def test_lichun_edge():
    # 2024年立春为2月4日：2月3日仍是癸卯年，2月5日已是甲辰年
    before = get_bazi(2024, 2, 3, 12, "男")
    after  = get_bazi(2024, 2, 5, 12, "男")
    assert before["年柱"]["天干"] == "癸"
    assert after["年柱"]["天干"]  == "甲"

# ── Case 4：夜子时边界 ────────────────────────────────────────────────────────
def test_yezishi_edge():
    # 1990-03-15 23:00（夜子时）日柱应等同于 1990-03-16 00:00（早子时）
    result_23 = get_bazi(1990, 3, 15, 23, "男")
    result_00 = get_bazi(1990, 3, 16,  0, "男")
    assert result_23["日柱"] == result_00["日柱"]
    assert result_23["时柱"] == result_00["时柱"]

# ── Case 5：闰月边界 ──────────────────────────────────────────────────────────
def test_runyue_edge():
    # 2023年闰二月（公历约3月22日—4月19日），验证不报错且结构完整
    result = get_bazi(2023, 4, 10, 12, "男")
    assert result["日元"] in list("甲乙丙丁戊己庚辛壬癸")
    assert all(k in result for k in ["年柱", "月柱", "日柱", "时柱", "日元", "性别"])

# ── Case 6：用户生日 2002-04-29 巳时（9:15取整为hour=9）────────────────────────
def test_user_birthday():
    result = get_bazi(2002, 4, 29, 9, "男")
    assert_valid_bazi(result)
    # 2002年为壬午年
    assert result["年柱"]["天干"] == "壬"
    assert result["年柱"]["地支"] == "午"
    # 4月29日在清明(4/5)之后、立夏(5/6)之前 → 月支为辰
    assert result["月柱"]["地支"] == "辰"
    # 9:00-11:00 为巳时
    assert result["时柱"]["地支"] == "巳"

# ── Case 7：元旦不换年柱（年柱在立春换，不在1月1日）─────────────────────────────
def test_yuandan_not_change_year_pillar():
    dec31 = get_bazi(2023, 12, 31, 12, "男")
    jan01 = get_bazi(2024,  1,  1, 12, "男")
    # 两天均在2024年立春（2月4日）之前，仍属癸卯年
    assert dec31["年柱"]["天干"] == "癸"
    assert jan01["年柱"]["天干"] == "癸"
    assert dec31["年柱"]["地支"] == "卯"
    assert jan01["年柱"]["地支"] == "卯"

# ── Case 8：公历闰年2月29日 ───────────────────────────────────────────────────
def test_gregorian_leap_feb29():
    # 2000年为庚辰年，2月29日在立春（2/4）之后，仍属庚辰年
    result = get_bazi(2000, 2, 29, 12, "男")
    assert_valid_bazi(result)
    assert result["年柱"]["天干"] == "庚"
    assert result["年柱"]["地支"] == "辰"

# ── Case 9：性别不影响四柱（只影响大运顺逆）─────────────────────────────────────
def test_gender_no_effect_on_four_pillars():
    male   = get_bazi(1990, 6, 15, 8, "男")
    female = get_bazi(1990, 6, 15, 8, "女")
    for pillar in ["年柱", "月柱", "日柱", "时柱"]:
        assert male[pillar] == female[pillar]
    assert male["日元"] == female["日元"]

# ── Case 10：子时/丑时边界（0:00是子时，1:00已是丑时）──────────────────────────
def test_zishi_choushift_boundary():
    result_0 = get_bazi(1995, 8, 20, 0, "男")
    result_1 = get_bazi(1995, 8, 20, 1, "男")
    assert result_0["时柱"]["地支"] == "子"   # 0:00 是早子时
    assert result_1["时柱"]["地支"] == "丑"   # 1:00 进入丑时

# ── Case 11：亥时（21:00-23:00）────────────────────────────────────────────────
def test_haishi():
    result = get_bazi(1995, 8, 20, 21, "男")
    assert result["时柱"]["地支"] == "亥"

# ── Case 12：夜子时跨月（月末23:00，日柱归次月1日）──────────────────────────────
def test_yezishi_month_boundary():
    # 1990-03-31 23:00 → 日柱用 1990-04-01 00:00
    result_31_23 = get_bazi(1990, 3, 31, 23, "男")
    result_apr01 = get_bazi(1990, 4,  1,  0, "男")
    assert result_31_23["日柱"] == result_apr01["日柱"]
