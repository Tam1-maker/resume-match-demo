import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "data" / "jobs_index.parquet"
META_PATH  = BASE_DIR / "data" / "skills_meta.json"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

INDEX_PATH = "../jobs_index_demo.parquet"
META_PATH  = "../skills_meta_demo.json"
RULES_PATH = "..\outputs_jobcluster_skillbundles_final\skill_bundles_rules.csv"

_SPLIT_PAT = re.compile(r"[,\|/;；、\n\r\t ]+")
PUNCT = re.compile(r"[：:，,。\.！!？?\(\)（）【】\[\]{}<>《》\"“”'’]")

# ====== 关键新增：福利/非技能黑名单（会用于岗位技能标签 & 简历抽取）======
BENEFIT_PHRASES = set("""
五险一金 六险一金 七险一金 五险 双休 周末双休 双休制 单休 大小周
带薪年假 年终奖 绩效奖金 全勤奖 加班补助 加班费 餐补 饭补 车补 房补 话补
补充医疗 商业保险 节假日福利 节日福利 团建 团建活动 下午茶
免费体检 体检 定期体检 员工旅游 旅游
包吃 包住 提供住宿 员工宿舍 班车 通勤班车
弹性工作 弹性办公 远程 远程办公 居家办公
股票期权 期权 股权 激励
""".split())

# 也会误入技能的高频“泛词”
GENERIC_NON_SKILL = set("""
福利 待遇 薪资 薪酬 工资 奖金 津贴 补贴 保险 公积金
发展空间 晋升 空间 氛围 环境 团队 年轻化
""".split())

STOP_PHRASES = set("""
姓名 民族 电话 手机 邮箱 出生 出生年月 年龄 籍贯
基本信息 教育背景 主修课程 实习经历 校园经历 项目经历 工作经历 荣誉奖项 技能证书 自我评价
负责 参与 协助 完成 推动 落地 沟通 协同 跨部门 组织 撰写 编写 输出 复盘
系统 平台 模块 功能 需求 文档 原型 流程图 交互 设计 开发 测试 上线 运维
能力 优秀 良好 具备 熟练 掌握 熟悉 了解 精通 擅长
一等奖 二等奖 三等奖 省级 国家级 校级 获奖 竞赛 证书
""".split())

ALIAS = {
    "py": "python",
    "python3": "python",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "powerbi": "power bi",
    "pbi": "power bi",
    "k8s": "kubernetes",
    "pyspark": "spark",
    "spark sql": "spark",
    "sqlserver": "sql server",
    "mssql": "sql server",
}

# ✅ 你已经扩充过，这里我只做了“范围拓宽”（同义表达/常见写法/英文token）
TRACK_TERMS = {
    "data_analytics": {
        "数据分析","数据可视化","报表","指标","bi","business intelligence","power bi","tableau",
        "统计","回归","sql","excel","分析模型","业务分析","商业分析","运营分析","用户分析","增长分析",
        "a/b","ab test","实验","漏斗","分群","留存","转化","cohort","roi","复盘","洞察","可视化"
    },
    "data_engineering": {
        "etl","elt","数据采集","数据抓取","爬虫","数据清洗","数据开发","数据工程","数仓","数据仓库",
        "数据建模","维度建模","dwd","dws","ads","ods","数据同步","数据集成","数据管道","pipeline",
        "spark","flink","hadoop","hive","kafka","hdfs","clickhouse","doris","presto","trino",
        "airflow","azkaban","oozie","dbt"
    },
    "data_science": {
        "数据科学","机器学习","深度学习","算法","算法工程师",
        "特征工程","模型训练","模型评估","预测","分类","聚类","推荐","排序","召回",
        "sklearn","scikit-learn","xgboost","lightgbm","catboost","pytorch","tensorflow","keras",
        "nlp","cv","llm","embedding","bert","transformer"
    },
    "data_governance": {
        "数据治理","数据质量","主数据","元数据","血缘","标签体系",
        "数据标准","指标体系","数据资产","数据安全","权限管理",
        "数据审计","口径","一致性","数据稽核","数据管理","mdm","meta","lineage"
    },
    "product": {
        "产品经理","产品助理","prd","需求","需求分析","需求文档",
        "原型","axure","墨刀","流程图","visio","产品文档","产品设计","用户调研",
        "项目管理","项目推进","里程碑","竞品","竞品分析","用户故事","roadmap","迭代"
    },
    "software": {
        "后端","服务端","api","接口","微服务","rpc",
        "java","spring","springboot","python","go","golang","c++","php","node","nodejs",
        "mysql","postgresql","postgre","redis","mongodb","elasticsearch","es",
        "linux","git","docker","kubernetes","k8s","nginx","jenkins","ci/cd"
    },
    "data_support": {
        "数据支持","数据专员","数据标注","数据处理","清洗校验","数据核对","数据录入",
        "测试","qa","功能测试","接口测试","质量检查","验收","抽检"
    },
    "platform_ops": {
        "运维","运维开发","devops","sre",
        "部署","监控","日志","告警","排障","应急","巡检",
        "jenkins","ci/cd","容器","云平台","服务器","堡垒机","prometheus","grafana"
    }
}

def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = s.replace("＋", "+").replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_token(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip()
    t = PUNCT.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def apply_alias(t: str) -> str:
    t = norm(t)
    return ALIAS.get(t, t)

def is_benefit_or_non_skill(t: str) -> bool:
    """✅ 新增：福利/非技能过滤（中英都尽量兜住）"""
    if not t:
        return True
    x = clean_token(str(t)).strip()
    if not x:
        return True
    if x in BENEFIT_PHRASES or x in GENERIC_NON_SKILL:
        return True

    # 包含型福利表达
    bad_sub = [
        "五险", "六险", "公积金", "带薪", "年终", "餐补", "房补", "车补", "话补", "补贴",
        "包吃", "包住", "宿舍", "班车", "团建", "下午茶", "体检", "旅游", "双休", "单休",
        "奖金", "提成", "福利", "待遇", "薪资", "薪酬", "五天", "弹性", "远程", "居家"
    ]
    for b in bad_sub:
        if b in x:
            return True
    return False

def is_noise_token(t: str) -> bool:
    if not t:
        return True
    x = t.strip()

    # ✅ 福利/非技能先杀掉
    if is_benefit_or_non_skill(x):
        return True

    if re.search(r"\d{6,}", x):  # 长数字：手机号/ID
        return True
    if re.search(r"20\d{2}\.\d{1,2}", x):  # 年月
        return True
    if re.search(r"gpa", x, re.I):
        return True
    if re.fullmatch(r"[\d\.\-/% ]+", x):  # 纯数字碎片
        return True
    if len(x) > 40:
        return True
    if x in STOP_PHRASES:
        return True
    return False

@st.cache_data(show_spinner=False)
def load_jobs_meta():
    jobs = pd.read_parquet(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return jobs, meta

# ---------- 简历结构化 ----------
def extract_major(text: str) -> str:
    """✅ 新增：通用专业抽取，不写死枚举"""
    if not isinstance(text, str):
        return ""

    # 1) 专业：xxx / 主修：xxx / 专业为xxx
    m = re.search(r"(专业|主修)[\s:：\-]{0,3}([^\n\r，,]{2,30})", text)
    if m:
        major = clean_token(m.group(2))
        return major[:20]

    # 2) xx大学 ... xx专业
    m = re.search(r"(大学|学院)[^\n\r]{0,12}([^\n\r，,]{2,20}专业)", text)
    if m:
        major = clean_token(m.group(2))
        return major[:20]

    # 3) 兜底：常见学科后缀（不枚举完整专业名）
    m = re.search(r"([^\n\r，,]{2,20}(工程|科学|技术|管理|经济|统计|数学|信息|计算机))", text)
    if m:
        return clean_token(m.group(1))[:20]

    return ""

def extract_profile(resume_text: str):
    text = resume_text if isinstance(resume_text, str) else ""
    profile = {}

    m = re.search(r"(本科|硕士|博士|研究生|专科)", text)
    if m:
        profile["学历"] = m.group(1)

    m = re.search(r"([^\s]{2,30}(大学|学院))", text)
    if m:
        profile["毕业院校"] = m.group(1)

    major = extract_major(text)
    if major:
        profile["专业"] = major

    m = re.search(r"籍\s*贯[:：]\s*([^\n\r]{2,20})", text)
    if m:
        profile["籍贯"] = clean_token(m.group(1))
    return profile

def extract_experiences(resume_text: str):
    text = resume_text if isinstance(resume_text, str) else ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    exps = []
    cur = {"time": "", "org": "", "role": "", "highlights": []}

    def flush():
        nonlocal cur
        if cur["org"] or cur["role"] or cur["highlights"]:
            cur["highlights"] = list(dict.fromkeys(cur["highlights"]))[:6]
            exps.append(cur)
        cur = {"time": "", "org": "", "role": "", "highlights": []}

    for l in lines:
        if re.search(r"20\d{2}\.\d{2}\s*-\s*20\d{2}\.\d{2}", l):
            flush()
            cur["time"] = l
            continue
        if ("公司" in l) and (len(l) <= 30):
            cur["org"] = l
            if "产品" in l:
                cur["role"] = "产品/软件开发"
            elif "爬虫" in l or "数据工程" in l:
                cur["role"] = "数据采集/数据工程"
            continue
        # 抓一些短要点（不改你原逻辑）
        if len(l) <= 60 and ("原型" in l or "流程图" in l or "接口" in l or "爬虫" in l or "数据库" in l or "清洗" in l or "入库" in l):
            cur["highlights"].append(l)
    flush()
    return exps

def extract_capabilities(resume_text: str, vocab: list, enable_fuzzy=True, cutoff=92):
    text = resume_text if isinstance(resume_text, str) else ""
    t_norm = norm(text)

    # ✅ 改造：不再用“少量seed”硬编码；改为“从词表+结构化token中抽”
    # 1) 先抽：简历里出现的英文token（可覆盖绝大多数工具/语言/框架）
    hard = set()
    tools = set()

    # 英文/符号token：python、c++、c#、.net、nodejs、sqlserver 等
    raw_tokens = [clean_token(x) for x in _SPLIT_PAT.split(t_norm) if x]
    raw_tokens = [apply_alias(x) for x in raw_tokens if x]

    # 过滤：只保留 ascii 且长度适中（工具/语言大多这样），并走 noise 规则
    ascii_tokens = []
    for x in raw_tokens:
        if not x:
            continue
        if not x.isascii():
            continue
        if len(x) < 2 or len(x) > 25:
            continue
        if is_noise_token(x):
            continue
        ascii_tokens.append(x)
    ascii_tokens = list(dict.fromkeys(ascii_tokens))[:1500]

    # 2) 用 vocab 做“精确命中 + 模糊补漏”（这是你原本的核心优势）
    vocab_set = set(vocab) if isinstance(vocab, list) else set()
    for tok in ascii_tokens[:800]:
        if tok in vocab_set:
            sk = tok
        else:
            sk = ""
            if enable_fuzzy and len(vocab_set) > 0:
                m = process.extractOne(tok, vocab, scorer=fuzz.ratio)
                if m and m[1] >= cutoff:
                    sk = m[0]

        if sk:
            sk = apply_alias(sk)
            # ✅ 工具 vs 硬技能：更通用的判别（不只 excel/ppt）
            if sk in {"excel","ppt","word","office","tableau","power bi","powerbi"}:
                tools.add("power bi" if sk == "powerbi" else sk)
            else:
                hard.add(sk)

    # 3) 中文“技能/工具”补充：从简历文本里按 TRACK_TERMS 扫一遍，扩宽覆盖
    #    （只作为补充，不影响你原本的 vocab 抽取逻辑）
    for _, terms in TRACK_TERMS.items():
        for term in terms:
            if term and (term.lower() in t_norm or term in text):
                # 把明显不是技能的“方向词”过滤掉（如 运营分析/业务分析仍可算方法域，不当硬技能）
                if any(k in term for k in ["分析", "管理", "治理", "测试", "运维", "回归", "统计", "增长", "商业"]):
                    continue
                tt = apply_alias(clean_token(term))
                if tt and not is_noise_token(tt) and len(tt) <= 25:
                    hard.add(tt)

    methods = set()
    domains = set()

    # ✅ 范围拓宽：从简历描述里抓“方法/领域”更鲁棒（不限定你个人简历）
    if any(k in text for k in ["数据挖掘", "挖掘"]):
        methods.add("数据挖掘")
    if any(k in text for k in ["数据分析", "可视化", "看板", "dashboard"]):
        methods.add("数据分析/可视化")
    if any(k in text for k in ["统计", "假设检验", "回归", "方差"]):
        methods.add("统计/建模基础")
    if any(k in text for k in ["机器学习", "深度学习", "特征工程", "模型"]):
        methods.add("机器学习/建模")

    if any(k in text for k in ["分布式", "大数据", "hadoop", "spark", "flink", "kafka"]):
        domains.add("分布式/大数据基础")
    if any(k in text for k in ["爬虫", "采集", "抓取", "scrapy", "selenium"]):
        domains.add("数据采集/爬虫")
    if any(k in text for k in ["数据库", "sql", "mysql", "postgresql", "redis", "mongodb"]):
        domains.add("数据库基础")
    if any(k in text for k in ["产品", "原型", "需求", "prd", "axure", "墨刀"]):
        domains.add("产品能力（原型/需求/文档）")
    if any(k in text for k in ["运维", "部署", "监控", "devops", "docker", "k8s", "kubernetes"]):
        domains.add("平台/工程化能力")

    # 方向计数（只做加分信号，不做硬筛）
    track_score = {k: 0 for k in TRACK_TERMS.keys()}
    for track, terms in TRACK_TERMS.items():
        for term in terms:
            if term and (term.lower() in t_norm or term in text):
                track_score[track] += 1

    return {
        "hard_skills": sorted(hard),
        "tools": sorted(tools),
        "methods": sorted(methods),
        "domains": sorted(domains),
        "track_score": track_score
    }

def extract_resume_all(resume_text: str, vocab: list, enable_fuzzy=True):
    return {
        "profile": extract_profile(resume_text),
        "capabilities": extract_capabilities(resume_text, vocab=vocab, enable_fuzzy=enable_fuzzy),
        "experiences": extract_experiences(resume_text),
    }

# ---------- 岗位侧 ----------
def build_job_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["职位名称"].fillna("").astype(str) + " " +
        df["技能标签"].fillna("").astype(str) + " " +
        df["岗位描述"].fillna("").astype(str)
    )

def job_track_score(job_text: str):
    t = norm(job_text)
    score = {k: 0 for k in TRACK_TERMS.keys()}
    for track, terms in TRACK_TERMS.items():
        for term in terms:
            if term and (term.lower() in t or term in job_text):
                score[track] += 1
    return score

def parse_job_skills(skill_text: str, vocab: list):
    """✅ 新增：岗位技能标签过滤福利词/非技能，并更宽松地按多分隔符切分"""
    if not isinstance(skill_text, str) or not skill_text.strip():
        return set()

    # 兼容：逗号/顿号/分号/竖线/空格等
    parts = [apply_alias(clean_token(x)) for x in _SPLIT_PAT.split(skill_text) if x.strip()]

    out = set()
    for p in parts:
        if not p:
            continue
        if is_noise_token(p):
            continue
        out.add(p)
    return out

# ---------- 评分（无阈值召回，改TopK召回） ----------
def compute_scores(resume_text: str, resume_struct: dict, jobs_df: pd.DataFrame, vocab: list,
                   w_text=0.70, w_skill=0.20, w_track=0.10):
    job_texts = build_job_text(jobs_df).tolist()

    corpus = [resume_text] + job_texts
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
    X = vec.fit_transform(corpus)
    text_sim = cosine_similarity(X[0:1], X[1:]).flatten()

    cand = set(resume_struct["capabilities"]["hard_skills"]) | set(resume_struct["capabilities"]["tools"])

    r_track = resume_struct["capabilities"]["track_score"]
    r_vec = np.array([r_track[k] for k in TRACK_TERMS.keys()], dtype=float)
    r_norm = np.linalg.norm(r_vec) + 1e-9

    skill_cov = np.zeros(len(jobs_df), dtype=float)
    track_sim = np.zeros(len(jobs_df), dtype=float)
    job_skill_sets = []

    for i, row in jobs_df.iterrows():
        js = parse_job_skills(row.get("技能标签", ""), vocab=vocab)
        job_skill_sets.append(js)

        if len(js) == 0:
            sc = 0.0
        else:
            sc = len(cand & js) / len(js)
        skill_cov[i] = sc

        jt = job_texts[i]
        j_track = job_track_score(jt)
        j_vec = np.array([j_track[k] for k in TRACK_TERMS.keys()], dtype=float)
        j_norm = np.linalg.norm(j_vec) + 1e-9
        track_sim[i] = float(np.dot(r_vec, j_vec) / (r_norm * j_norm))

    raw_final = w_text * text_sim + w_skill * skill_cov + w_track * track_sim

    mn, mx = float(raw_final.min()), float(raw_final.max())
    norm_final = (raw_final - mn) / (mx - mn + 1e-9)

    out = jobs_df.copy()
    out["text_sim"] = text_sim
    out["skill_cov_job"] = skill_cov
    out["track_sim"] = track_sim
    out["raw_final"] = raw_final
    out["final"] = norm_final
    out["_job_skill_set"] = job_skill_sets
    return out

def plot_bar(res_df: pd.DataFrame, topk=15):
    df = res_df.sort_values("final", ascending=True).tail(min(topk, len(res_df)))
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = (df["职位名称"].astype(str) + " | " + df["公司"].astype(str)).tolist()
    ax.barh(labels, df["final"].values)
    ax.set_xlabel("匹配度（final，归一化）")
    ax.set_title("Top岗位匹配度（条形图）")
    st.pyplot(fig)

def explain_overlap(resume_struct: dict, job_skill_set: set):
    cand = set(resume_struct["capabilities"]["hard_skills"]) | set(resume_struct["capabilities"]["tools"])
    inter = sorted(list(cand & job_skill_set))
    missing = sorted(list(job_skill_set - cand))
    return inter, missing

# =========================
# UI
# =========================
st.set_page_config(page_title="TopK召回：岗位出现更多", layout="wide")
st.title("简历画像/能力/经历抽取 + TopK召回（岗位出现更多）")

jobs_df, meta = load_jobs_meta()
vocab = meta.get("skills_vocab", [])

with st.sidebar:
    st.header("数据状态")
    st.success(f"岗位数：{len(jobs_df)}")
    st.success(f"技能词表：{len(vocab)}")

    st.header("筛选条件")
    cities = sorted([c for c in jobs_df["城市"].dropna().astype(str).unique().tolist() if c.strip()])
    city_sel = st.multiselect("城市（可多选）", options=cities, default=[])

    exp_options = sorted([x for x in jobs_df["经验"].dropna().astype(str).unique().tolist() if x.strip()])
    edu_options = sorted([x for x in jobs_df["学历"].dropna().astype(str).unique().tolist() if x.strip()])
    exp_sel = st.multiselect("经验（可多选）", options=exp_options, default=[])
    edu_sel = st.multiselect("学历（可多选）", options=edu_options, default=[])

    recall_k = st.slider("召回岗位TopK（越大岗位出现越多）", 100, 5000, 1200, 100)
    topn = st.slider("最终展示TopN", 10, 100, 30, 5)

    st.header("权重")
    w_text = st.slider("文本相似度权重", 0.0, 1.0, 0.70, 0.05)
    w_skill = st.slider("岗位技能覆盖权重", 0.0, 1.0, 0.20, 0.05)
    w_track = st.slider("方向一致性权重", 0.0, 1.0, 0.10, 0.05)
    s = w_text + w_skill + w_track
    w_text, w_skill, w_track = w_text/s, w_skill/s, w_track/s

    enable_fuzzy = st.checkbox("启用英文技能模糊匹配补漏", value=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("粘贴简历全文")
    resume_text = st.text_area("直接粘贴简历全文即可", height=340)
    run_btn = st.button("开始抽取 + 匹配", type="primary")

with col2:
    st.subheader("抽取结果预览（画像/能力/经历）")
    if run_btn and resume_text.strip():
        resume_struct = extract_resume_all(resume_text, vocab=vocab, enable_fuzzy=enable_fuzzy)

        st.markdown("### 画像")
        st.json(resume_struct["profile"], expanded=False)

        st.markdown("### 核心能力标签")
        cap = resume_struct["capabilities"]
        st.write("硬技能：", ", ".join(cap["hard_skills"]) if cap["hard_skills"] else "—")
        st.write("工具：", ", ".join(cap["tools"]) if cap["tools"] else "—")
        st.write("方法：", "；".join(cap["methods"]) if cap["methods"] else "—")
        st.write("领域：", "；".join(cap["domains"]) if cap["domains"] else "—")
        st.write("方向信号：", cap["track_score"])

        st.markdown("### 经历摘要")
        if resume_struct["experiences"]:
            for e in resume_struct["experiences"]:
                st.write(f"- **{e.get('time','')}** | {e.get('org','')} | {e.get('role','')}")
                for h in e.get("highlights", [])[:4]:
                    st.write(f"  - {h}")
        else:
            st.write("—")

st.divider()

if run_btn and resume_text.strip():
    df = jobs_df.copy()
    if city_sel:
        df = df[df["城市"].astype(str).isin(city_sel)]
    if exp_sel:
        df = df[df["经验"].astype(str).isin(exp_sel)]
    if edu_sel:
        df = df[df["学历"].astype(str).isin(edu_sel)]

    st.write(f"筛选后岗位数：{len(df)}")
    if len(df) == 0:
        st.warning("筛选后岗位为 0，请放宽筛选条件。")
        st.stop()

    matched = compute_scores(
        resume_text=resume_text,
        resume_struct=resume_struct,
        jobs_df=df.reset_index(drop=True),
        vocab=vocab,
        w_text=w_text, w_skill=w_skill, w_track=w_track
    )

    recall_df = matched.sort_values("final", ascending=False).head(min(recall_k, len(matched)))
    show_df = recall_df.head(topn)

    st.subheader("召回结果（TopK）")
    st.dataframe(
        show_df[["final","raw_final","text_sim","skill_cov_job","track_sim","城市","职位名称","公司","经验","学历","salary_mid"]],
        use_container_width=True
    )

    st.subheader("Top岗位匹配度条形图（归一化final）")
    plot_bar(show_df, topk=15)

    st.markdown("### 逐条岗位解释（命中能力/缺口）")
    for _, row in show_df.iterrows():
        with st.expander(f"{row.get('职位名称','')} | {row.get('公司','')} | final={row['final']:.2f}"):
            st.write(f"**分数拆解：** text={row['text_sim']:.2f} | cov_job={row['skill_cov_job']:.2f} | track={row['track_sim']:.2f} | raw={row['raw_final']:.3f}")
            st.write(f"**城市/经验/学历：** {row.get('城市','')} / {row.get('经验','')} / {row.get('学历','')}")
            st.write(f"**技能标签：** {row.get('技能标签','')}")
            desc = row.get("岗位描述","")
            st.write(f"**岗位描述（截断）：** {str(desc)[:260]}…")

            job_sk = row["_job_skill_set"] if isinstance(row["_job_skill_set"], set) else set(row["_job_skill_set"])
            inter, missing = explain_overlap(resume_struct, job_sk)
            st.write("**命中能力（硬技能+工具）：**", ", ".join(inter[:25]) if inter else "无")
            st.write("**缺口（岗位技能-你现有）：**", ", ".join(missing[:25]) if missing else "无")
else:
    st.info("粘贴简历后点击「开始抽取 + 匹配」。")

