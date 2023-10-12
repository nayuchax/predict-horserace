import requests
from bs4 import BeautifulSoup


# 馬の名前の取得
def name_getter(url: str) -> str:
    res = requests.get(url)

    # 文字化け防止
    res.encoding = res.apparent_encoding

    soup = BeautifulSoup(res.text, "html.parser")
    horse_name = soup.select_one("div.horse_title > h1").string

    return horse_name


# テーブルデータの取得
def table_getter(url: str) -> list[list]:
    res = requests.get(url)

    # 文字化け防止
    res.encoding = res.apparent_encoding

    soup = BeautifulSoup(res.text, "html.parser")
    table_obj = soup.find("table", {"class", "db_h_race_results nk_tb_common"})
    tr_obj = table_obj.find_all("tr")

    table_data = []
    for tr_elements in tr_obj:
        item_list = []
        for item in tr_elements:
            d = item.text
            if d != "\n":
                item_list.append(d)
        table_data.append(item_list)

    return table_data
