"""Independent demand-style cases; authored by Codex, not real-user logs or gold labels."""
import json
from pathlib import Path

SEARCH=[
 "弱光图像增强时如何抑制噪声放大，并保留边缘细节？",
 "多帧图像去噪怎样处理运动物体造成的重影？",
 "针对遮挡行人的检测方案，哪些技术改进涉及特征融合？",
 "低功耗终端如何进行关键词语音唤醒？",
 "锂电池热失控早期预警通常使用哪些传感信号？",
 "电池组液冷结构怎样减少电芯之间的温差？",
 "光伏组件表面的灰尘如何自动清除？",
 "电动汽车充电过程中有哪些绝缘检测方案？",
 "工业机械臂怎样利用视觉完成抓取位置定位？",
 "仓储机器人怎样避开动态障碍物？",
 "高压阀门如何降低密封部位的泄漏？",
 "换热器怎样在紧凑结构下提高换热效率？",
 "有哪些利用膜分离处理含油废水的技术路线？",
 "吸附材料如何实现对重金属离子的选择性去除？",
 "血细胞检测过程中如何提高样本保存稳定性？",
 "微流控芯片如何进行样本混合与分离？",
 "柔性压力传感器怎样提高灵敏度并降低迟滞？",
 "光纤传感系统怎样补偿温度引起的测量误差？",
 "农作物病害识别如何应对复杂背景与光照变化？",
 "建筑墙体保温结构如何兼顾防火与施工便捷性？",
]
FILTER=[
 ("图像去噪",{"ipc_subclass":"G06T","year_from":2020}),
 ("电池散热",{"ipc_section":"H","patent_type":"发明申请"}),
 ("阀门密封",{"ipc_subclass":"F16K","year_from":2018,"year_to":2022}),
 ("图像增强",{"city":"南京市","year_from":2019}),
 ("机器人定位",{"city":"深圳市","patent_type":"发明申请"}),
 ("废水处理",{"ipc_section":"C","year_from":2021}),
 ("光纤传感",{"year_from":2015,"year_to":2017}),
 ("换热结构",{"patent_type":"实用新型","year_from":2020}),
 ("语音唤醒",{"ipc_section":"G","city":"北京市"}),
 ("传感器标定",{"year_from":2024,"year_to":2024}),
 ("图像去噪",{"city":"不存在的城市_验收"}),
 ("电池散热",{"year_from":2025,"year_to":2026}),
 ("膜分离",{"ipc_subclass":"B01D","patent_type":"发明授权"}),
 ("激光测量",{"city":"合肥市","year_from":2018,"year_to":2021}),
 ("微流控",{"city":"上海市","patent_type":"发明申请"}),
 ("行人检测",{"ipc_section":"G","ipc_subclass":"G06T"}),
 ("风机降噪",{"year_from":2023,"patent_type":"实用新型"}),
 ("温度补偿",{"city":"杭州市","year_to":2020}),
 ("锂电池",{"year_from":2024,"year_to":2018}),
 ("图像去噪",{"ipc_section":"H","ipc_subclass":"G06T"}),
]
COMPARE=[
 "比较所选专利的实现方法，分别给出摘要依据。",
 "比较所选方案的关键组成，摘要没有披露的部分请明确标记。",
 "比较两件专利解决的技术问题，区分相同点和各自关注点。",
 "只比较两种图像处理方案的数据处理步骤，不评价法律保护范围。",
 "比较方案涉及的传感信号与处理方法，不能推断未披露的测量精度。",
 "比较所选散热方案的流道或换热结构，只使用摘要明确内容。",
 "归纳所选技术方案的核心方法，每件专利最多一条结论。",
 "逐项核对所选摘要中明确出现的性能数值，并列出原句。",
 "比较所选专利是否明确披露温度补偿，只能说提及或摘要未提及。",
 "比较所选候选的技术路径，说明哪些信息仍需要阅读完整说明书。",
]
BOUNDARY=[
 "判断这个产品是否侵权。", "分析这两件专利的保护范围。", "查询这些专利的法律状态。",
 "这些技术是否具有新颖性？", "请判断这项方案的创造性。", "查询这件专利的申请人名称。",
 "这个产品会不会侵权？", "给出这件专利的诉讼记录。", "评估实施这个方案的侵权风险。",
 "判断所选技术是否构成侵权。",
]


def main():
    rows=[]
    def add(kind,q,filters,policy):
        rows.append({"case_id":f"W{len(rows)+1:03}","kind":kind,"question":q,"filters":filters,"expected_policy":policy,
                     "provenance":"independently authored by Codex; not collected real-user queries", "relevance_gold":None,"label_status":"pending_review"})
    for q in SEARCH:add("search",q,{},"Return candidates with exact source evidence; relevance requires review")
    for q,f in FILTER:add("filter",q,f,"Reject inconsistent filters; otherwise preserve all hard constraints and never widen an empty search")
    for q in COMPARE:add("analysis",q,{},"Select actual retrieved patents; every accepted factual claim must reference its own canonical abstract")
    for q in BOUNDARY:add("boundary",q,{},"Explicitly decline unsupported legal or applicant identity determination")
    target=Path(__file__).with_name("acceptance_cases.jsonl")
    target.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    print(f"Wrote {len(rows)} independent cases; semantic relevance labels are pending")


if __name__=="__main__":main()
