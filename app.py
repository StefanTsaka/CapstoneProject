import os, json, re, time, datetime as dt
import pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ––––– CONFIG –––––
SEARCH_MODEL = "gpt-4o-mini-search-preview"   # web-search capable
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"

NEEDED   = 10    # rows in final table
ASK_FOR  = 20    # ask for extras
MAX_TRY  = 6     # prompt retries
API_RETRY= 4
SLEEP    = 2     # sec back-off

MISSION = (
    "The San Francisco-Marin Food Bank's mission is to end hunger in San Francisco and Marin by distributing food and addressing the root causes of hunger. We envision a just and equitable society that nurtures a resilient community. We envision a community where everyone will have access to nutritious food of their choosing and will be uplifted by a network of support. We believe that food is a basic human right."
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ––––– HELPERS –––––
def retry(fn):
    def wrap(*a, **k):
        for i in range(API_RETRY):
            try: return fn(*a, **k)
            except openai.RateLimitError: time.sleep(SLEEP*(i+1))
        st.error("OpenAI rate-limit — try later."); st.stop()
    return wrap

@retry
def ask(model, msgs):                 # chat wrapper
    return openai.chat.completions.create(model=model, messages=msgs)

@retry
def emb(text):                        # embedding wrapper
    return openai.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding

# ––––– 1 · FETCH & DEDUP –––––
def fetch_unique():
    prompt = (
        f"search: Provide {ASK_FOR} CURRENT US grant opportunities (Apply-Now link included) "
        f"for nonprofits in high-school education, youth equity, or college readiness. "
        f"Exclude any grant whose deadline is before {dt.date.today()}. "
        "Return ONLY a JSON array. Keys: title, sponsor, amount, deadline (YYYY-MM-DD "
        "or 'rolling'), url (direct apply link), summary."
    )
    titles_seen, urls_seen, rows = set(), set(), []
    for _ in range(MAX_TRY):
        raw = ask(SEARCH_MODEL, [{"role":"user","content":prompt}]).choices[0].message.content
        try: data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", raw, re.S); data = json.loads(m.group()) if m else []

        today = dt.date.today()
        for d in data:
            g = {k: d.get(k, "N/A") for k in
                 ("title","sponsor","amount","deadline","url","summary")}
            # future deadline check
            future = True
            dl = g["deadline"].lower()
            if dl != "rolling":
                try:  future = dt.datetime.strptime(dl[:10],"%Y-%m-%d").date() >= today
                except: future = False
            # de-dupe by title OR url
            tkey = g["title"].strip().lower(); ukey = g["url"].strip().lower()
            if future and tkey not in titles_seen and ukey not in urls_seen:
                rows.append(g); titles_seen.add(tkey); urls_seen.add(ukey)
            if len(rows) == NEEDED: return rows
    return rows   # may be <10 after retries

# ––––– 2 · RANK & ADD “WHY” –––––
def make_table(raw):
    df = pd.DataFrame(raw)
    base_vec = emb(MISSION)
    df["Match%"] = (df.summary.apply(lambda s:
        cosine_similarity([emb(s)], [base_vec])[0][0]*100).round(1))
    df = df.sort_values("Match%", ascending=False).reset_index(drop=True)

    whys=[]
    for _, r in df.iterrows():
        q = (f'In one sentence: why does the grant "{r.title}" align with '
             f'the mission "{MISSION}"?')
        whys.append(ask(CHAT_MODEL,[{"role":"user","content":q}])
                     .choices[0].message.content.strip())
    df["Why It Fits"] = whys

    # reorder & rename columns
    return df[["title","Match%","amount","deadline",
               "sponsor","summary","url","Why It Fits"]].rename(columns={
        "title":"Title","amount":"Amount","deadline":"Deadline",
        "sponsor":"Sponsor","summary":"Grant Summary","url":"URL"
    })

# ––––– STREAMLIT APP –––––
st.title("Food Bank Capstone Project (Grant Matcher for SF Food Bank – GPT-4o Search)")
st.write("> **Mission:**", MISSION)

if st.button("Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT-4o searching web and compiling unique future-deadline grants …"):
        grants = fetch_unique()
        if len(grants) < NEEDED:
            st.error(f"Only found {len(grants)} unique future-deadline grants. Click again.")
        else:
            st.session_state["tbl"] = make_table(grants)
            st.success("Table ready!")

if "tbl" in st.session_state:
    st.dataframe(st.session_state["tbl"], use_container_width=True)
else:
    st.caption("Press the rocket to generate your ranked list.")
